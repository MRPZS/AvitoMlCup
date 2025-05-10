import polars as pl
import pandas as pd
import numpy as np
from datetime import timedelta
import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

# Определение директории с данными
DATA_DIR = 'data/'

# Загрузка данных
print("Загрузка данных...")
df_test_users = pl.read_parquet(f'{DATA_DIR}/test_users.pq')
df_clickstream = pl.read_parquet(f'{DATA_DIR}/clickstream.pq')
df_cat_features = pl.read_parquet(f'{DATA_DIR}/cat_features.pq')
df_text_features = pl.read_parquet(f'{DATA_DIR}/text_features.pq')
df_event = pl.read_parquet(f'{DATA_DIR}/events.pq')

# Константы
EVAL_DAYS_THRESHOLD = 14

# Разделение данных на обучающую и тестовую выборки по времени
threshold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_THRESHOLD)
df_train = df_clickstream.filter(df_clickstream['event_date'] <= threshold)
df_eval = df_clickstream.filter(df_clickstream['event_date'] > threshold)[['cookie', 'node', 'event']]

# Подготовка данных для валидации
# Оставляем только контактные события и только те пары (cookie, node), которых нет в обучающей выборке
df_eval = df_eval.join(df_train.select(['cookie', 'node']).unique(), on=['cookie', 'node'], how='anti')
df_eval = df_eval.filter(
    pl.col('event').is_in(
        df_event.filter(pl.col('is_contact') == 1)['event'].unique()
    )
)
df_eval = df_eval.filter(
    pl.col('cookie').is_in(df_train['cookie'].unique())
).filter(
    pl.col('node').is_in(df_train['node'].unique())
)
df_eval = df_eval.unique(['cookie', 'node'])

# Функция для расчета метрики Recall@k
def recall_at(df_true, df_pred, k=40):
    return df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
        how='left',
        on=['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum() / pl.col('value').count()
        ]
    )['value'].mean()

# Функция для получения популярных товаров
def get_popular(df, top_n=40):
    popular_nodes = df.group_by('node').agg(pl.col('cookie').count()).sort('cookie', descending=True).head(top_n)['node'].to_list()
    return popular_nodes

# Функции-помощники, вынесенные за пределы класса для корректной передачи в пул процессов
def create_candidates_for_user(user, user_history_dict, all_nodes, top_popular_nodes, n_candidates=100):
    """Создает кандидатов для одного пользователя."""
    # Находим исторические node пользователя
    user_nodes = user_history_dict.get(user, [])
    
    # Находим node, с которыми пользователь еще не взаимодействовал
    candidate_nodes = [node for node in all_nodes if node not in user_nodes]
    
    # Если кандидатов слишком много, оставляем популярные и случайные
    if len(candidate_nodes) > n_candidates:
        # Оставляем все популярные nodes из топ популярных, если они еще не встречались пользователю
        popular_candidates = [node for node in top_popular_nodes if node in candidate_nodes]
        
        # Оставшиеся кандидаты выбираем случайно
        remaining_candidates = [node for node in candidate_nodes if node not in popular_candidates]
        np.random.shuffle(remaining_candidates)
        
        # Объединяем популярные и случайные кандидаты
        final_candidates = popular_candidates + remaining_candidates[:n_candidates - len(popular_candidates)]
    else:
        final_candidates = candidate_nodes
    
    # Возвращаем пары (пользователь, node)
    return [{"cookie": user, "node": node} for node in final_candidates]

def process_user_batch(user_batch, user_history_dict, all_nodes, top_popular_nodes, n_candidates):
    """Обрабатывает батч пользователей."""
    batch_candidates = []
    for user in user_batch:
        user_candidates = create_candidates_for_user(
            user, 
            user_history_dict, 
            all_nodes, 
            top_popular_nodes, 
            n_candidates
        )
        batch_candidates.extend(user_candidates)
    return batch_candidates

# Класс для создания признаков на основе взаимодействий пользователя
class FeatureEngineering:
    def __init__(self, df_clickstream, df_cat_features, df_event):
        self.df_clickstream = df_clickstream
        self.df_cat_features = df_cat_features
        self.df_event = df_event

    def process_clickstream(self):
        """Создание признаков на основе истории взаимодействий пользователя."""
        # Добавляем информацию о типе события (контактное или нет)
        df_clickstream_with_event_type = self.df_clickstream.join(
            self.df_event, on="event", how="left"
        )
        
        # Агрегируем данные по пользователям
        user_features = df_clickstream_with_event_type.group_by("cookie").agg([
            pl.col("item").n_unique().alias("unique_items_count"),
            pl.col("node").n_unique().alias("unique_nodes_count"),
            pl.col("event").count().alias("total_events"),
            pl.col("is_contact").sum().alias("contact_events"),
            pl.col("surface").n_unique().alias("unique_surfaces"),
            pl.col("platform").n_unique().alias("unique_platforms"),
            pl.col("event_date").max().alias("last_activity"),
            pl.col("event_date").min().alias("first_activity"),
        ])
        
        # Добавляем признак активности пользователя (в днях)
        user_features = user_features.with_columns([
            (pl.col("last_activity") - pl.col("first_activity")).dt.total_days().alias("activity_days")
        ])
        
        # Добавляем признак частоты контактов
        user_features = user_features.with_columns([
            (pl.col("contact_events") / pl.col("total_events")).fill_null(0).alias("contact_rate")
        ])
        
        # Топ категорий для каждого пользователя
        user_categories = df_clickstream_with_event_type.join(
            self.df_cat_features.select(["item", "category"]), on="item", how="left"
        ).group_by(["cookie", "category"]).agg([
            pl.len().alias("category_count")
        ]).sort(["cookie", "category_count"], descending=[False, True])
        
        # Для каждого пользователя берем топ-3 категории
        top_categories = user_categories.group_by("cookie").head(3)
        
        # Преобразуем в широкий формат
        top_categories_pivot = top_categories.pivot(
            index="cookie", 
            on="category",
            values="category_count"
        ).fill_null(0)
        
    # Здесь реализованы остальные методы класса, как в вашем основном коде
    def create_candidates(self, users, all_nodes, top_popular_nodes, n_candidates=100, chunksize=None, max_workers=None):
        """
        Создает кандидатов для каждого пользователя с использованием улучшенной параллельной обработки.
        
        Args:
            users: список пользователей
            all_nodes: список всех доступных node
            top_popular_nodes: список популярных node
            n_candidates: количество кандидатов для каждого пользователя
            chunksize: размер батча для одного процесса (автоматически рассчитывается, если None)
            max_workers: максимальное количество рабочих процессов (по умолчанию cpu_count)
        
        Returns:
            DataFrame с кандидатами
        """
        print("    Создание кандидатов для пользователей...")
        
        # Получаем исторические взаимодействия пользователя и преобразуем в словарь для быстрого доступа
        user_history = self.df_clickstream.filter(pl.col("cookie").is_in(users)).group_by("cookie").agg([
            pl.col("node").unique().alias("history_nodes")
        ])
        
        # Создаем словарь для быстрого доступа к истории каждого пользователя
        user_history_dict = {row['cookie']: row['history_nodes'] for row in user_history.iter_rows(named=True)}
        
        # Определяем оптимальное количество процессов и размер батча
        num_cpu = multiprocessing.cpu_count()
        max_workers = max_workers or num_cpu
        
        # Определяем оптимальный размер батча (по умолчанию 10 пользователей на 1 процесс)
        if chunksize is None:
            # Идеальный размер чанка - баланс между накладными расходами на создание процессов
            # и эффективным использованием многоядерности
            total_users = len(users)
            if total_users <= max_workers:
                chunksize = 2  # по одному пользователю на процесс, если пользователей мало
            else:
                chunksize = max(1, min(10, total_users // max_workers))
        
        print(f"    Запуск в {max_workers} процессах с размером батча {chunksize}...")
        
        # Разбиваем пользователей на батчи по chunksize
        user_batches = [users[i:i + chunksize] for i in range(0, len(users), chunksize)]
        
        # Создаем частичную функцию с предварительно заданными параметрами
        process_batch_func = partial(
            process_user_batch, 
            user_history_dict=user_history_dict,
            all_nodes=all_nodes,
            top_popular_nodes=top_popular_nodes,
            n_candidates=n_candidates
        )
        
        # Улучшенная параллельная обработка с прогресс-баром
        all_candidates = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи
            futures = {executor.submit(process_batch_func, batch): i for i, batch in enumerate(user_batches)}
            
            # Отслеживаем прогресс
            with tqdm.tqdm(total=len(user_batches), desc="    Обработка батчей") as pbar:
                for future in as_completed(futures):
                    # Обрабатываем результаты по мере их поступления
                    try:
                        batch_candidates = future.result()
                        all_candidates.extend(batch_candidates)
                    except Exception as e:
                        batch_idx = futures[future]
                        print(f"    Ошибка в батче {batch_idx}: {e}")
                    pbar.update(1)
        
        print(f"    Всего создано {len(all_candidates)} кандидатов для {len(users)} пользователей")
        return pl.DataFrame(all_candidates)
    
    def process_item_features(self):
        """Создание признаков для товаров."""
        # Агрегируем данные по node
        node_features = self.df_clickstream.group_by("node").agg([
            pl.col("item").n_unique().alias("unique_items_in_node"),
            pl.col("cookie").n_unique().alias("unique_users_viewed"),
            pl.col("event").count().alias("total_node_views")
        ])
        
        # Добавляем признаки категории для node
        node_category = self.df_cat_features.join(
            self.df_clickstream.select(["item", "node"]).unique(), on="item", how="inner"
        ).group_by("node").agg([
            pl.col("category").mode().alias("main_category"),
            pl.col("location").mode().alias("main_location")
        ])
        
        # Объединяем признаки node
        all_node_features = node_features.join(node_category, on="node", how="left")
        
        return all_node_features

    def create_interaction_matrix(self):
        """Создает матрицу взаимодействий пользователь-node."""
        # Подсчет количества взаимодействий
        interaction_counts = self.df_clickstream.group_by(["cookie", "node"]).agg([
            pl.len().alias("interaction_count")
        ])
        
        return interaction_counts

def build_features_for_model(candidates, user_features, node_features, interaction_matrix):
    """Строит признаки для градиентного бустинга."""
    
    # Объединяем кандидатов с признаками пользователей и товаров
    candidates_with_features = candidates.join(user_features, on="cookie", how="left")
    candidates_with_features = candidates_with_features.join(node_features, on="node", how="left")
    
    # Добавляем количество взаимодействий (если есть)
    candidates_with_features = candidates_with_features.join(
        interaction_matrix, on=["cookie", "node"], how="left"
    ).with_columns([
        pl.col("interaction_count").fill_null(0)
    ])
    
    return candidates_with_features

def prepare_training_data(df_clickstream, df_cat_features, df_event, threshold_date):
    """Подготовка данных для обучения модели."""
    print("  Разделение данных на обучающую и валидационную выборки...")
    # Разделяем данные на обучающую и валидационную выборки
    df_train_history = df_clickstream.filter(df_clickstream['event_date'] <= threshold_date)
    df_valid_targets = df_clickstream.filter(df_clickstream['event_date'] > threshold_date)
    
    print("  Создание положительных примеров...")
    # Получаем контактные события для создания положительных примеров
    contact_events = df_event.filter(pl.col('is_contact') == 1)['event'].unique().to_list()
    
    # Создаем положительные примеры (пользователь взаимодействовал с node)
    positive_examples = df_valid_targets.filter(
        pl.col('event').is_in(contact_events)
    ).select(['cookie', 'node']).unique()
    
    # Добавляем метку класса
    positive_examples = positive_examples.with_columns([
        pl.lit(1).alias("target")
    ])
    
    print("  Создание инженерных признаков...")
    # Создаем инженерные признаки
    feature_engineering = FeatureEngineering(df_train_history, df_cat_features, df_event)
    
    print("  Обработка данных о пользователях...")
    user_features = feature_engineering.process_clickstream()
    
    print("  Обработка данных о товарах...")
    node_features = feature_engineering.process_item_features()
    
    print("  Создание матрицы взаимодействий...")
    interaction_matrix = feature_engineering.create_interaction_matrix()
    
    # Получаем список пользователей и node для генерации отрицательных примеров
    users = positive_examples['cookie'].unique().to_list()
    all_nodes = df_train_history['node'].unique().to_list()
    
    print("  Получение популярных товаров...")
    # Получаем популярные node
    popular_nodes = get_popular(df_train_history, top_n=100)
    
    print("  Создание кандидатов для отрицательных примеров...")
    # Создаем кандидатов для отрицательных примеров
    candidates = feature_engineering.create_candidates(users, all_nodes, popular_nodes, n_candidates=200)
    
    print("  Исключение положительных примеров из кандидатов...")
    # Исключаем положительные примеры из кандидатов
    negative_examples = candidates.join(
        positive_examples.select(['cookie', 'node']), 
        on=['cookie', 'node'], 
        how='anti'
    ).with_columns([
        pl.lit(0).alias("target")
    ])
    
    print("  Объединение положительных и отрицательных примеров...")
    # Объединяем положительные и отрицательные примеры
    training_data = pl.concat([positive_examples, negative_examples])
    
    print("  Добавление признаков для обучения модели...")
    # Добавляем признаки для обучения модели
    training_data_with_features = build_features_for_model(
        training_data, user_features, node_features, interaction_matrix
    )
    
    print(f"  Подготовлено {len(training_data_with_features)} примеров для обучения")
    return training_data_with_features, user_features, node_features, interaction_matrix

def train_gradient_boosting_model(training_data):
    """Обучает модель градиентного бустинга."""
    # Преобразуем polars DataFrame в pandas DataFrame для LightGBM
    print("  Преобразование данных для LightGBM...")
    training_data_pd = training_data.to_pandas()
    
    # Определяем признаки и целевую переменную
    features = [col for col in training_data_pd.columns if col not in ['cookie', 'node', 'target']]
    X = training_data_pd[features]
    y = training_data_pd['target']
    
    print(f"  Количество признаков: {len(features)}")
    print(f"  Количество примеров: {len(y)}")
    print(f"  Распределение целевой переменной: {y.value_counts().to_dict()}")
    
    # Проверка на наличие пропущенных значений
    print("  Проверка на пропущенные значения...")
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"  Обнаружено {missing_values} пропущенных значений. Заполняем нулями.")
        X = X.fillna(0)
    
    # Разделяем данные на обучающую и проверочную выборки
    print("  Разделение данных на обучающую и проверочную выборки...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создаем датасеты LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Параметры модели
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Обучаем модель
    print("  Обучение модели...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Оцениваем модель
    print("  Оценка модели на валидационной выборке...")
    y_pred = gbm.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(f"  Validation AUC: {auc:.4f}")
    
    # Визуализируем важность признаков
    print("  Создание визуализации важности признаков...")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(gbm, max_num_features=20)
    plt.title(f"Feature Importance (AUC: {auc:.4f})")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("  Сохранен график важности признаков (feature_importance.png)")
    
    return gbm, features

def generate_recommendations(df_test_users, df_train, df_cat_features, df_event, model, feature_names, user_features, node_features, interaction_matrix):
    """Генерирует рекомендации для тестовых пользователей."""
    test_users = df_test_users['cookie'].unique().to_list()
    all_nodes = df_train['node'].unique().to_list()
    
    # Получаем популярные node
    popular_nodes = get_popular(df_train, top_n=100)
    
    # Создаем инженерные признаки для тестовых пользователей
    feature_engineering = FeatureEngineering(df_train, df_cat_features, df_event)
    
    # Создаем кандидатов для тестовых пользователей
    test_candidates = feature_engineering.create_candidates(test_users, all_nodes, popular_nodes, n_candidates=1000)
    
    # Добавляем признаки для тестовых кандидатов
    test_candidates_with_features = build_features_for_model(
        test_candidates, user_features, node_features, interaction_matrix
    )
    
    # Преобразуем в pandas DataFrame
    test_data_pd = test_candidates_with_features.to_pandas()
    
    # Убеждаемся, что у нас есть все нужные признаки
    for feature in feature_names:
        if feature not in test_data_pd.columns:
            test_data_pd[feature] = 0
    
    # Предсказываем вероятности
    test_data_pd['score'] = model.predict(test_data_pd[feature_names])
    
    # Преобразуем обратно в polars DataFrame
    predictions = pl.from_pandas(test_data_pd[['cookie', 'node', 'score']])
    
    # Сортируем предсказания по пользователю и скору
    predictions = predictions.sort(['cookie', 'score'], descending=[False, True])
    
    # Для каждого пользователя оставляем топ-40 рекомендаций
    recommendations = predictions.group_by('cookie').head(40).select(['cookie', 'node'])
    
    return recommendations

def main():
    """Основная функция для запуска рекомендательной системы."""
    print("Запуск рекомендательной системы с градиентным бустингом...")
    
    # Разделение данных на обучающую и тестовую выборки по времени
    threshold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_THRESHOLD)
    df_train = df_clickstream.filter(df_clickstream['event_date'] <= threshold)
    
    # Подготовка данных для обучения
    print("Подготовка данных для обучения...")
    training_data, user_features, node_features, interaction_matrix = prepare_training_data(
        df_clickstream, df_cat_features, df_event, threshold
    )
    
    # Обучение модели
    model, feature_names = train_gradient_boosting_model(training_data)
    
    # Генерация рекомендаций для тестовых пользователей
    print("Генерация рекомендаций...")
    recommendations = generate_recommendations(
        df_test_users, df_train, df_cat_features, df_event, 
        model, feature_names, user_features, node_features, interaction_matrix
    )
    
    # Оценка качества рекомендаций на валидационной выборке
    print("Оценка качества рекомендаций...")
    recall = recall_at(df_eval, recommendations, k=40)
    print(f"Recall@40: {recall:.4f}")
    
    # Сохранение рекомендаций в CSV
    print("Сохранение рекомендаций...")
    recommendations.write_csv("recommendations.csv", include_header=True)
    print("Рекомендации сохранены в файл recommendations.csv")
    
    # Сравнение с базовым решением (популярные товары)
    popular_recommendations = get_popular(df_train)
    df_pred_pop = pl.DataFrame({
        'node': [popular_recommendations for _ in range(len(df_test_users))], 
        'cookie': df_test_users['cookie'].to_list()
    })
    df_pred_pop = df_pred_pop.explode('node')
    popular_recall = recall_at(df_eval, df_pred_pop, k=40)
    print(f"Baseline Recall@40 (Popular Items): {popular_recall:.4f}")
    
    return recommendations

if __name__ == "__main__":
    main()
