import pandas as pd 
from dotenv import load_dotenv

import openai
import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm 
from collections import Counter
from pprint import pprint 
import os
from torch.nn.parallel import DistributedDataParallel
class LargeScaleEvaluator(nn.Module):
    def __init__(self,model,item_title_dict,item_genre_dict,tokenizer,rank,args,alpha = .5,split='test'):
        super().__init__()
        self.model = model
        self.item_title_dict = item_title_dict
        self.item_genre_dict = item_genre_dict
        #lower case all the genres in the dict
        for k,v in item_genre_dict.items():
            item_genre_dict[k] = [x.lower().replace('-', ' ') for x in v]
        self.tokenizer = tokenizer
        self.rank = rank
        counts = Counter(sum([v for v in item_genre_dict.values()],[]))

        #keep counts if above 200 
        
        self.counts = {k:v for k,v in counts.items() if v > 100}
        #sort self.counds for the print 
        self.counts = dict(sorted(self.counts.items(), key=lambda item: item[1],reverse=True))
        print(f"{len(self.counts)=}")
        pprint(f"{self.counts=}")

        self.genre_list = list(self.counts.keys())


        # if args.data_name == 'goodbooks':
        self.genre_list = [x.lower().replace('-', ' ') for x in self.genre_list]
        
        self.genre_set = ', '.join(self.genre_list)
        self.args = args

        print(f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv')
        if os.path.exists(f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv'):
            print(f"{f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv'=}")
            self.df = pd.read_csv(f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv')



        self.alpha = alpha 
        self.alpha2 = None

    def set_alpha2(self,alpha2):
        self.alpha2 = alpha2   
    def getGenreDeltaBatch(self, s1,s2,labels,topk,genre_1 , genre_2,rank ,neg=False,small = None,proportional = False):
        if self.args.mask_control_labels:
            labels = torch.zeros_like(labels)

        # Determine whether the model is using DistributedDataParallel
        with torch.no_grad():
            if self.args.embedding_module not in ['RecVAEGenreVAE', 'GenreTEARS']:
                if isinstance(self.model, DistributedDataParallel):
                    # Use the batched generate_recommendations_batch function
                    topk1 = self.model.module.generate_recommendations_batch(s1, self.tokenizer, labels, topk, rank, alpha=self.alpha, neg=neg)
                    topk2 = self.model.module.generate_recommendations_batch(s2, self.tokenizer, labels, topk, rank, alpha=self.alpha2 if self.alpha2 is not None else self.alpha, neg=neg)
                else:
                    topk1 = self.model.generate_recommendations_batch(s1, self.tokenizer, labels, topk, rank, alpha=self.alpha, neg=neg)
                    topk2 = self.model.generate_recommendations_batch(s2, self.tokenizer, labels, topk, rank, alpha=self.alpha2 if self.alpha2 is not None else self.alpha, neg=neg)
            else:
                if isinstance(self.model, DistributedDataParallel):
                    topk1 = self.model.module.generate_recommendations_batch(topk=topk, rank=rank, alpha=self.alpha, neg=neg, small=small,proportional = proportional)
                    topk2 = self.model.module.generate_recommendations_batch(mask_genre=genre_2, topk=topk, rank=rank, alpha=self.alpha2 if self.alpha2 is not None else self.alpha, neg=neg, small=small, proportional = proportional)
                else:
                    topk1 = self.model.generate_recommendations_batch(data_tensor=labels, fav_genre = genre_1,topk=topk, rank=rank, alpha=self.alpha, neg=neg, small=small, proportional = proportional)
                    topk2 = self.model.generate_recommendations_batch(data_tensor=labels,fav_genre = genre_1, mask_genre=genre_2, topk=topk, rank=rank, alpha=self.alpha2 if self.alpha2 is not None else self.alpha, neg=neg, small=small, proportional = proportional)

        # Map the top-k indices to movie titles and genres



       # Assuming topk1 and topk2 are lists of sublists
        movie_titles1 = [[self.item_title_dict[i] for i in sublist] for sublist in topk1]
        movie_titles2 = [[self.item_title_dict[i] for i in sublist] for sublist in topk2]

        movie_genres1 = [[self.item_genre_dict[i] for i in sublist] for sublist in topk1]
        movie_genres2 = [[self.item_genre_dict[i] for i in sublist] for sublist in topk2]

        # Initialize lists to hold the NDCG changes
        change_up_list = []
        change_down_list = []

        # Iterate through each user's recommendations
        for i in range(len(movie_genres1)):
            change_up = self.genrewise_ndcg(movie_genres1[i], genre_2[i], min_k=0, max_k=topk) - self.genrewise_ndcg(movie_genres2[i], genre_2[i], min_k=0, max_k=topk)



            change_up_list.append(change_up)


            
            change_down = self.genrewise_ndcg(movie_genres1[i], genre_1[i], min_k=0, max_k=topk) - self.genrewise_ndcg(movie_genres2[i], genre_1[i], min_k=0, max_k=topk)
            change_down_list.append(change_down)
        # return change_up_list

        
             
        

        return change_down_list, change_up_list, movie_titles1, movie_titles2

        
    def getGenreDelta(self, s1,s2,labels,topk,genre_1 , genre_2,rank ,neg=False,small = False):
        if self.args.mask_control_labels:
            labels = torch.zeros_like(labels)


        if  self.args.embedding_module not in  ['RecVAEGenreVAE','GenreTEARS']:
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg = neg)
                topk2 = self.model.module.generate_recommendations(s2, self.tokenizer, labels, 

                                                                   topk, rank,alpha = self.alpha2 if self.alpha2 is not None else self.alpha,neg = neg)

            else:
                topk1 = self.model.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg =neg,small = True)

                
                topk2 = self.model.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha2 if self.alpha2 is not None else self.alpha,neg = neg,small = True)




        else: 
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(  topk =topk, rank=rank,alpha = self.alpha,neg = neg,small = small)
                topk2 = self.model.module.generate_recommendations(mask_genre = genre_2,fav_genre = genre_1, topk = topk, rank =rank,alpha = self.alpha2 if self.alpha2 is not None else self.alpha,neg = neg,small = small)

            else:
                topk1 = self.model.generate_recommendations( data_tensor = labels,topk =topk, rank=rank,alpha = self.alpha,neg = neg,small = small)
                topk2 = self.model.generate_recommendations(data_tensor = labels,mask_genre = genre_2, fav_genre = genre_1,topk = topk, rank =rank,alpha = self.alpha2 if self.alpha2 is not None else self.alpha,neg = neg,small = small)
       
        movie_titles1 = [self.item_title_dict[i] for i in topk1]


        movie_titles2 = [self.item_title_dict[i] for i in topk2]



        # print(f"{movie_titles2=}")

        movie_genres1 = [self.item_genre_dict[i] for i in topk1]
        movie_genres2 = [self.item_genre_dict[i] for i in topk2]



        change_up =  self.genrewise_ndcg(movie_genres1,genre_2,min_k = 0,max_k = topk) - self.genrewise_ndcg(movie_genres2,genre_2,min_k = 0,max_k = topk)

        base_val = self.genrewise_ndcg(movie_genres1,genre_2,min_k = 0,max_k = topk)
        changed_val = self.genrewise_ndcg(movie_genres2,genre_2,min_k = 0,max_k = topk)

        change_down = self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk) - self.genrewise_ndcg(movie_genres2,genre_1,min_k = 0,max_k = topk)
        
        
        return change_down,change_up,movie_titles1,movie_titles2
    
    def genrewise_ndcg_b(self, genre_movies_batch, genre, min_k=0, max_k=None):
    # genre_movies_batch is now a list of lists, where each sublist corresponds to the genres of movies for a single summary
        ndcg_values = []



        for genre_movies in genre_movies_batch:
            # Calculate relevance for the specified genre
            relevance = [print(genre) for g in genre_movies[min_k:max_k]]
            # relevance = [1 if genre in g else 0 for g in genre_movies[min_k:max_k]]

            # Compute DCG
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

            # Compute ideal DCG (IDCG)
            idcg = sum(1 / np.log2(i + 2) for i in range(len(relevance)))

            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_values.append(ndcg)

        # Return the list of NDCG values for each summary in the batch
        return ndcg_values

    def genrewise_ndcg(self,genre_movies, genre, min_k=0, max_k=None):
        
        relevance = [1 if genre in g   else 0 for g in genre_movies[min_k:max_k]]
        # print(genre,f"{relevance=}")

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(1 / np.log2(i + 2) for i in range(len(relevance)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return ndcg
     
    def print_ranking_diffences(self,titles1,titles2): 
        for i,(title1, title2) in enumerate(zip(titles1, titles2)):
            print(f"{i} {title1:<80} {title2}")

    def promptGPT(self,k_sample, prompts, labels, seed =2024,split='val'):
    
        np.random.seed(seed)
        # self.model.eval()
        load_dotenv()

        self.genre_list = [x.lower().replace('-', ' ') for x in self.genre_list]

        print(f"{self.genre_list=}")
        c = 0
        # self.args.llm_backbone = 'gpt-4o'
        print(f"{self.args.llm_backbone=}")
        key = os.getenv("OPEN-AI-SECRET") 
        # key = os.getenv("OPEN-AI-SECRET") if 'gpt' in self.args.llm_backbone else os.getenv("LLAMA-KEY")
        openai.api_key = key 
        # # openai.api_base = None
        # if 'gpt' not in  self.args.llm_backbone : 
        #     print(f"{self.args.llm_backbone=}")
        #     openai.api_base = "https://api.llama-api.com" 

        print(f"{openai.api_base=}")

        move_up_genres =[]
        move_down_genres = []
        outputs = []
        keys = []
        deltas_ndcg_up = []
        deltas_ndcg_down = []
        deltas_ndcg_up = []
        deltas_ndcg_down = []
        if os.path.exists(f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv'):
            df = pd.read_csv(f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv')
        else:
            df = pd.DataFrame({'keys':[]})


        for k in (pbar:=tqdm(k_sample)): 
            if k  in df['keys'].values:
                continue
            s = prompts[k]
            labs = labels[k]
                
            prompt = \
                        f"""
                        You are a professional editor please identify the users preferred genres from the following:
                        {self.genre_set}
                        """

            user_prompt =\
                        f"""
                        Please identify the users most favorite genre from the following summary and the least favorite genre: 
                        in the format Favorite: [genre]\n Least Favorite: [genre]
                        {s}.
                        Remember the genre you pick must be in this set of genres {self.genre_set} and in the format Favorite: [genre]\n Least Favorite: [genre]
                        """
                        
            msg = [
                        {
                            "role": "system",
                            "content": prompt

                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]


            genres = openai.ChatCompletion.create(
                            model=self.args.llm_backbone,
                            # model='gpt-4o',
                            messages=msg,
                            max_tokens=300,
                            temperature=0.001,
                            seed=2024,
                        )['choices'][0]['message']['content']
                        
            lines = genres.split('\n')
            try:
                favorite_genre = lines[0].split(': ')[1].lower().replace('-', ' ')
                least_favorite_genre = lines[1].split(': ')[1].lower().replace('-', ' ')
            except:
                continue
            if favorite_genre not in self.genre_list or least_favorite_genre not in self.genre_list:
                c+= 1
                pbar.set_description(f"pass {c} {favorite_genre} {least_favorite_genre}")    
                
                continue
            
            msg = [
                        {
                            "role": "system",
                            "content": prompt

                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {'role': 'assistant',
                        'content': genres},
                        {'role':'user',
                        'content':
                        f'Now using this setup write the a new summary in the same style that reflects that {favorite_genre} is your least favorite\
                            and {least_favorite_genre} is your favorite only output the full summary keep the format and length the same' }
                    ]

            gpt_output = openai.ChatCompletion.create(
                            model=self.args.llm_backbone,
                            # model='gpt-4o',
                            messages=msg,
                            max_tokens=300,
                            temperature=0.001,
                            seed=2024,
                        )['choices'][0]['message']['content']


            delta1,delta2,movies1,movies2 = self.getGenreDelta(s,gpt_output,labs,20,favorite_genre,least_favorite_genre,rank = self.rank)
            move_down_genres.append(favorite_genre)
            move_up_genres.append(least_favorite_genre)
            outputs.append(gpt_output)  
            keys.append(k)
            deltas_ndcg_down.append(delta1)
            deltas_ndcg_up.append(delta2)

            
            pbar.set_description(f"gen {favorite_genre}, average ndcgs up {np.mean(deltas_ndcg_up)} average ndcg down {np.mean(deltas_ndcg_down)}")    

            

            data = {
                'move_down_genres': move_down_genres,
                'move_up_genres': move_up_genres,
                'outputs': outputs,
                'keys': keys,
                'deltas_ndcg_down': deltas_ndcg_down,
                'deltas_ndcg_up': deltas_ndcg_up,
            }


        new_data = pd.DataFrame(data)
        df = pd.concat([df,new_data])
        df.to_csv(f'./saved_user_summary/{self.args.data_name}/gpt4_results_large_genre_{split}_{self.args.llm_backbone}.csv')
        print('WROTE DF')
        self.df = df
            
    def evaluate_genre_b(self, model, dataloader, prompts, topk, rank=None, dir='more', neg=False, return_arr=False, genre_changed=None):
        model.eval()

        if genre_changed is not None:
            genres = [genre_changed]
        else:
            # Get the top 10 genres from self.counts (already sorted)
            self.counts = dict(sorted(self.counts.items(), key=lambda item: item[1], reverse=True))
            genres = [x for x in list(self.counts.keys())[:10]]

        self.delta_up = []
        self.delta_down = []

        small = True if self.args.embedding_module in ['RecVAEGenreVAE', 'GenreTEARS'] else False
        item = 'books' if self.args.data_name == 'goodbooks' else 'movies'

        out_metrics = {}
        
        for b in tqdm(dataloader, desc='Controllability'):
            uids = b['idx'].flatten().tolist()
            labels = b['labels_tr'].to(rank)
            mean_per_user = []

            for genre in genres:
                

                genre_up = []
                batch_s1 = []
                batch_s2 = []

                for uid, label in zip(uids, labels):
                    # Create genre-specific summaries
                    rep_s1 = f'Summary: {genre} {item} are the users favourite they really enjoy {genre} content '


                    rep_s2 = f''

                    batch_s1.append(rep_s1)
                    batch_s2.append(rep_s2)
                    

                genre_list = [genre] * len(batch_s1)
                genre_dummie = ['action'] * len(batch_s1)

                # Batched getGenreDelta
                down, up, movies1, movies2 = self.getGenreDeltaBatch(batch_s2, batch_s1, labels, topk, genre_dummie, genre_list, rank=rank, neg=neg,small = small)                           
                # down,up,movies1,movies2 = self.getGenreDelta('',rep_s,label,topk,'action',genre ,rank =rank,neg=neg,small = small)

                # Filter out None values in up and down
                self.delta_up.append(up)
                self.delta_down.append(down)

                genre_up.append(up)

                mean_per_user.append(down)

                out_metrics[genre] = np.mean(genre_up)

                # print(f"{genre} {np.mean(genre_up)=}")

                
            tqdm.write(f"Average NDCGs up: {np.mean(self.delta_up)} | {genre}: {np.mean(genre_up)}")

        if return_arr:
            return self.delta_up, self.delta_down
        else:
            return out_metrics, np.mean(self.delta_down)

    def evaluate_b(self, dataloader, prompts, topk, rank=None, max_delta=False,proportional = False):
        self.model.eval()

        # Initialize lists to track deltas and genres
        self.delta_up = []
        self.delta_down = []
        self.least_fav = []
        self.most_fav = []

        # Progress bar to monitor controlability
        for b in tqdm(dataloader, desc='Controllability'):
            uids = b['idx'].flatten().tolist()  # Batch of user IDs
            labels = b['labels_tr'].to(rank)  # Batch of labels for users

            # Filter the users in the batch who are present in the DataFrame
            valid_uids = [uid for uid in uids if uid in self.df['keys'].values]
            valid_indices = [uids.index(uid) for uid in valid_uids]
            valid_labels = labels[valid_indices]

            # Collect prompts for the valid users
            valid_prompts = [prompts[uid] for uid in valid_uids]
            print(f"{len(valid_prompts)=}")


            # Batch process all valid users in the batch
            if valid_uids:
                sub_df = self.df[self.df['keys'].isin(valid_uids)]
                gpt_outputs = sub_df['outputs'].values.tolist()
                genre1_list = sub_df['move_down_genres'].apply(lambda x: x.lower().replace('-', ' ')).values
                genre2_list = sub_df['move_up_genres'].apply(lambda x: x.lower().replace('-', ' ')).values

                self.least_fav.extend(genre2_list)
                self.most_fav.extend(genre1_list)
                
                # Get the genre deltas in a batched manner
                down, up, movies1, movies2 = (self.getMaxDelta(valid_prompts, gpt_outputs, valid_labels, topk, genre1_list, genre2_list, rank=rank)
                                            if max_delta else
                                            self.getGenreDeltaBatch(valid_prompts, gpt_outputs, valid_labels, topk, genre1_list, genre2_list, rank=rank,proportional = proportional))

                # Append the batch of deltas


                self.delta_up.extend(up)
                self.delta_down.extend(down)

            # Display running averages


            avg_up = np.mean(self.delta_up) if self.delta_up else 0
            avg_down = np.mean(self.delta_down) if self.delta_down else 0
            tqdm.write(f"Average NDCG up: {avg_up} | Average NDCG down: {avg_down}")

        # Final averages
        final_avg_up = np.mean(self.delta_up) if self.delta_up else 0
        final_avg_down = np.mean(self.delta_down) if self.delta_down else 0
        tqdm.write(f"Final average NDCG up: {final_avg_up} | Final average NDCG down: {final_avg_down}")

        # Return the final averages
        return final_avg_up, final_avg_down

    def evaluate(self,dataloader,prompts,topk,rank = None,max_delta = False):
        self.model.eval()



        # df = pd.read_csv(f'./results/{self.args.data_name}/gpt4_results_large_genre.csv')
        self.delta_up = []
        self.delta_down = []
        # print(self.model.device)
        # print(rank)
        # exit()
        c = 0
       

        for b in (pbar:=tqdm(dataloader,desc = 'Controlability')):
            uids = b['idx'].flatten().tolist()
            #if rank is not set set the rank
            labels = b['labels_tr'].to(rank)
            self.least_fav = []
            self.most_fav = []   


            for uid,label in zip(uids,labels):
                if uid in self.df['keys'].values:


                    sub_df = self.df[self.df['keys'] == uid]
                    gpt_output = sub_df['outputs'].values[0]
                    genre1 = sub_df['move_down_genres'].values[0].lower().replace('-', ' ')
                    genre2 = sub_df['move_up_genres'].values[0].lower().replace('-', ' ')
                    self.least_fav.append(genre2)
                    self.most_fav.append(genre1)
                        
                        
                    down,up,movies1,movies2 = self.getGenreDelta(prompts[uid],gpt_output,label,topk,genre1 ,genre2 ,rank =rank) if not max_delta else self.getMaxDelta(prompts[uid],gpt_output,label,topk,genre1 ,genre2 ,rank =rank)

                    self.delta_up.append(up)
                    # print(f"{self.delta_up=}")
                    self.delta_down.append(down)
               


            print(f"average ndcgs up {np.mean(self.delta_up)} average ndcg down {np.mean(self.delta_down)}")
        pbar.set_description(f"average ndcgs up {np.mean(self.delta_up)} average ndcg down {np.mean(self.delta_down)}")

        # raise Exception
        return np.mean(self.delta_up),np.mean(self.delta_down)
    
    def evaluate_genre(self,model,dataloader,prompts,topk,rank = None,dir = 'more',neg = False,return_arr = False,genre_changed = None):
            model.eval()
            # print(self.genre_set)
            # raise Exception
            if genre_changed is not None: 
                genres = [genre_changed]
            else:
                #get the top 10 genres from self.counds
                #make sure self.counts is sorted so we get the ten most popular genres
                self.counts = dict(sorted(self.counts.items(), key=lambda item: item[1],reverse=True))
                genres = [x for x in list(self.counts.keys())[:10]]
            # df = pd.read_csv(f'./results/{self.args.data_name}/gpt4_results_large_genre.csv')
            self.delta_up = []
            self.delta_down = []
            # print(self.model.device)
            # print(rank)
            # exit()
            if self.args.embedding_module in ['RecVAEGenreVAE','GenreTEARS']:
                small = True
            else:
                small = False
            
            if self.args.data_name == 'goodbooks':
                item = 'books'
            else:
                item ='movies'
            out_metrics = {}
            for b in (pbar:=tqdm(dataloader,desc = 'Controlability')):
                uids = b['idx'].flatten().tolist()
                #if rank is not set set the rank
                
                labels = b['labels_tr'].to(rank)
                mean_per_user = []
                for genre in genres:



                    if genre == 'sport':
                        continue
                    genre_up = []
                    for uid,label in zip(uids,labels):
                    
                        rep_s = f'Summary: {genre} {item} are the users favourite they really enjoy {genre}tic moments '
                        rep_s = f'Summary: {genre} {item} are the users favourite they really enjoy {genre} content '
                        # rep_s = f'Summary: {genre}'

                        down,up,movies1,movies2 = self.getGenreDelta('',rep_s,label,topk,'action',genre ,rank =rank,neg=neg,small = small)

                        #for up filter out None values 

                        # if up is not None:


                        self.delta_up.append(up)
                        # print(f"{delta_up=}")
                        self.delta_down.append(down)

                        
                        genre_up.append(up)
                        mean_per_user.append(down)


                    out_metrics[genre] = np.mean(genre_up)





                pbar.set_description(f"average ndcgs up {np.mean(self.delta_up)} average ndcg up {genre} {np.mean(genre_up)}")
                # return self.delta_up
            if return_arr:
                
                return self.delta_up,self.delta_up
            else: 
                return out_metrics,np.mean(self.delta_down)
                        

            
    def getMaxDelta(self, s1,s2,labels,topk,genre_1 , genre_2,rank ,neg=False):

        # print(f"{s1=}")
        # print(f"{s2=}")
        if self.args.mask_control_labels:
        # if True:
            labels = torch.zeros_like(labels)
        # labels = torch.zeros_like(labels)
        # print(f"{labels.sum()=}")
        
        if  self.args.embedding_module not in  ['RecVAEGenreVAE','GenreTEARS']:
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg = neg,small=False)
                topk2 = self.model.module.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg,small=False)

            else:
                topk1 = self.model.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg =neg,small=False)
                
                topk2 = self.model.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg,small=False)
        else: 
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(  topk =topk, rank=rank,alpha = self.alpha,neg = neg,small=False)
                topk2 = self.model.module.generate_recommendations(mask_genre = genre_2, topk = topk, rank =rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg,small=False)

            else:
                topk1 = self.model.generate_recommendations( data_tensor = labels,topk =topk, rank=rank,alpha = self.alpha,neg = neg,small=False)

                
                topk2 = self.model.generate_recommendations(data_tensor = labels,mask_genre = genre_2, topk = topk, rank =rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg,small=False)


        movie_titles1 = [self.item_title_dict[i] for i in topk1]
        movie_titles2 = [self.item_title_dict[i] for i in topk2]
        # print(f"{movie_titles1=}")
        # print(f"{movie_titles2=}")

        movie_genres1 = [self.item_genre_dict[i] for i in topk1]
        movie_genres2 = [self.item_genre_dict[i] for i in topk2]
        
        change_down = self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk) - 0
        # print(f"{genre_1=}")
        # print(f"{self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk)=}")
        # print(f"{change_down=}")
        # print(f"{self.genrewise_ndcg(movie_genres2,genre_1,min_k = 0,max_k = topk)=}")
        change_up =  self.genrewise_ndcg(movie_genres1,genre_2,min_k = 0,max_k = topk) - 1
        # print(f"{genre_2=}")
        # print(f"{change_up=}")
        # raise Exception
        
        return change_down,change_up,movie_titles1,movie_titles2

        
    def get_qualitative_metrics(self,):
        return {'fav_genres': self.most_fav, 'least_fav_genres': self.least_fav,'down': self.delta_down, 'up': self.delta_up}