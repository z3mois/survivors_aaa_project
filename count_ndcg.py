'''
Возможно, что-то будет бажить, будьте аккуратны
'''

from tqdm import tqdm
from sklearn.metrics import ndcg_score

def count_ndcg_for_all_resumes(model, df, k=None):
  '''
  ранжируем к вакансии вообще все резюме из кластера 
  это походит на боевые условия, но имхо, ждать тут высокого ndcg не стоит из-за разреженнности векторов
  '''
  if not k:
    k = df.shape[0]

  df = df.drop_duplicates(['vac_des', 'res_des'])
  df.index = range(df.shape[0])
  if df.shape[0] > 0:
    ndcg_vals = []
    for vac in tqdm(df['vac_des'].unique()):
      possible_res = df[df['vac_des'] == vac][['res_des', 'rank']]
      if possible_res.shape[0] < 3:
        continue

      true_rank_vector = np.zeros((df.shape[0]))
      true_rank_vector[possible_res.index] = possible_res['rank']

      model_preds = model_inference_dataset(model, df)

      if len(true_rank_vector) < k:
        ndcg_vals.append(ndcg_score([true_rank_vector], [model_preds]))
      else:
        ndcg_vals.append(ndcg_score([true_rank_vector[:k]], [model_preds[:k]]))

    return ndcg_vals
  
def count_ndcg_for_appropriate_resumes(model, df, k=None):
  '''
  ранжируем только подходящие к вакансии резюме
  то есть делаем предикт только для них и смотрим, как алгоритм умеет сортировать "хорошие" сэмплы
  '''
  if not k:
    k = float('inf')

  df = df.drop_duplicates(['vac_des', 'res_des'])
  df.index = range(df.shape[0])
  if df.shape[0] > 0:
    ndcg_vals = []
    for vac in tqdm(df['vac_des'].unique()):
      possible_res = df[df['vac_des'] == vac][['City', 'microcat_name', 'vac_des', 'res_des', 'rank']]
      if possible_res.shape[0] < 3:
        continue

      true_rank_vector = possible_res['rank'].values
      model_probs = model_inference_dataset(model, possible_res)

      if len(true_rank_vector) < k:
        ndcg_vals.append(ndcg_score([true_rank_vector], [model_probs]))
      else:
        ndcg_vals.append(ndcg_score([true_rank_vector[:k]], [model_probs[:k]]))

    return ndcg_vals
