from RatingSystem import RatingSystem
import numpy as np
import math

class MySystemPearson(RatingSystem):
    def __init__(self):
        super().__init__()
        self.TOP_K = 8
        self.MIN_OVERLAP = 4.0
        self.BAYES_C = 5.0
        
        self.global_mean = 2.5
        self.movie_bayesian_avgs = {}
        self.user_avgs = {}
        self.user_centered_ratings = {}
        self.user_norms = {}
        self.movie_to_users = {} 
        
        self._precompute()

    def _precompute(self):
        all_sum = 0.0
        all_count = 0
        
        for ratings in self.movie_ratings.values():
            if ratings:
                all_sum += sum(ratings)
                all_count += len(ratings)
        self.global_mean = all_sum / all_count if all_count > 0 else 2.5

        for m_id, ratings in self.movie_ratings.items():
            n = len(ratings)
            if n > 0:
                self.movie_bayesian_avgs[m_id] = (sum(ratings) + self.BAYES_C * self.global_mean) / (n + self.BAYES_C)
        
        for user_id, user_obj in self.users.items():
            ratings = user_obj.ratings
            if not ratings:
                continue
                
            avg = sum(ratings.values()) / len(ratings)
            self.user_avgs[user_id] = avg
            
            centered = {m: r - avg for m, r in ratings.items()}
            self.user_centered_ratings[user_id] = centered
            self.user_norms[user_id] = math.sqrt(sum(c * c for c in centered.values()))
            
            for m, c in centered.items():
                if m not in self.movie_to_users:
                    self.movie_to_users[m] = {}
                self.movie_to_users[m][user_id] = c

    def rate(self, user, movie_id):
        fallback_pred = self.movie_bayesian_avgs.get(movie_id, self.global_mean)
        
        if movie_id not in self.movie_to_users or not user.ratings:
            return math.floor(float(np.clip(fallback_pred, 0.5, 5.0)) * 2.0 + 0.5) / 2.0
            
        target_avg = sum(user.ratings.values()) / len(user.ratings)
        target_centered = {m: r - target_avg for m, r in user.ratings.items()}
        target_norm = math.sqrt(sum(c * c for c in target_centered.values()))
        
        if target_norm == 0:
            return math.floor(float(np.clip(target_avg, 0.5, 5.0)) * 2.0 + 0.5) / 2.0
            
        neighbors = self.movie_to_users[movie_id]
        similarities = []
        
        for neighbor_id, neighbor_movie_centered_rating in neighbors.items():
            neighbor_norm = self.user_norms.get(neighbor_id, 0)
            if neighbor_norm == 0:
                continue
                
            neighbor_centered = self.user_centered_ratings[neighbor_id]
            dot_product = 0.0
            overlap_count = 0
            
            if len(target_centered) < len(neighbor_centered):
                for m, c in target_centered.items():
                    if m in neighbor_centered:
                        dot_product += c * neighbor_centered[m]
                        overlap_count += 1
            else:
                for m, c in neighbor_centered.items():
                    if m in target_centered:
                        dot_product += target_centered[m] * c
                        overlap_count += 1
                        
            raw_sim = dot_product / (target_norm * neighbor_norm)
            significance_weight = min(overlap_count, self.MIN_OVERLAP) / self.MIN_OVERLAP
            sim = raw_sim * significance_weight
            
            if sim > 0:
                similarities.append((sim, neighbor_movie_centered_rating))
                
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:self.TOP_K]
        
        if not top_k:
            return math.floor(float(np.clip(fallback_pred, 0.5, 5.0)) * 2.0 + 0.5) / 2.0
        
        votes = {x / 2.0: 0.0 for x in range(1, 11)}
        
        for sim, centered_rating in top_k:
            pred_from_neighbor = target_avg + centered_rating
            bucket = math.floor(float(np.clip(pred_from_neighbor, 0.5, 5.0)) * 2.0 + 0.5) / 2.0
            votes[bucket] += sim
            
        best_rating = max(votes, key=votes.get)
        return best_rating

    def __str__(self):
        return 'MySystem 155294 and 155877'
