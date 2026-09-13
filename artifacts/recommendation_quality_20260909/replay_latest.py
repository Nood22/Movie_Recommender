import sys,json,sqlite3,re
from pathlib import Path
sys.path.insert(0,'/home/mila/a/adls/tears_project_final')
import torch
from pilot_recommender import PilotHybridRecommender
from tears_preference_ranking import align_scores,explicit_genre_preferences

torch.set_num_threads(2)
with sqlite3.connect('file:logs/pilot_study.sqlite3?mode=ro',uri=True) as db:
 t,r,p=db.execute("SELECT recorded_at,request_id,payload_json FROM events WHERE system='TEARS' AND event_type='recommendation_result' ORDER BY rowid DESC LIMIT 1").fetchone()
d=json.loads(p); e=d['effective_model_input'];q=d['request']
m=PilotHybridRecommender(device='cpu')
profiles={'backend':e['summary'],'display':e['display_summary']}
enc=m.tokenizer(list(profiles.values()),padding='max_length',truncation=True,max_length=512,return_tensors='pt')
with torch.inference_mode(): logits,_=m.tears(enc.input_ids,enc.attention_mask)
g={genre:torch.tensor([genre in str(v).split('|') for v in m.catalog.genres]) for genre in m.genre_names}
out={'recorded_at':t,'request_id':r,'model':m.status()['models']['tears'],'profiles':profiles,'logged_top12':[i['title'] for i in d['recommendations']],'experiments':[]}
for ix,(name,summary) in enumerate(profiles.items()):
 for aligned in (False,True):
  prefs=explicit_genre_preferences(summary)
  scores=align_scores(logits[ix:ix+1],g,prefs) if aligned else logits[ix:ix+1]
  for exclusions,year,label in [(q['excluded_movie_ids']+q['liked_movie_ids'],2015,'current'),(q['liked_movie_ids'],2015,'selected_only'),(q['liked_movie_ids'],None,'selected_only_all_years')]:
   items=m._ranked_items(scores,exclusions,12,year)
   out['experiments'].append({'profile':name,'aligned':aligned,'candidate_policy':label,'preferences':prefs,'items':[{k:i[k] for k in ('movie_id','title','genres','score')} for i in items]})
rows=m.catalog
out['catalog_items']=len(rows)
years=rows.title.str.extract(r'\((\d{4})\)\s*$',expand=False).fillna('0').astype(int)
out['items_2015_plus']=int((years>=2015).sum())
out['eligible_current']=int(((years>=2015)&~rows.movieId.isin(q['excluded_movie_ids']+q['liked_movie_ids'])).sum())
path=Path('artifacts/recommendation_quality_20260909');path.mkdir(exist_ok=True)
(path/'latest_request_ablation.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out['experiments']: print(x['profile'],x['aligned'],x['candidate_policy'],[i['title'] for i in x['items']][:6])
print({k:out[k] for k in ('catalog_items','items_2015_plus','eligible_current')})
