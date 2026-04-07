from RatingSystem import RatingSystem
import numpy as np

class MySystemBaseline(RatingSystem):
    def __init__(self):
        super().__init__()

        self.global_mean = 0.0
        self.movie_averages = {}
        self.user_deviations = {}

        self._calculate_baselines()

    def _calculate_baselines(self):
        all_ratings_sum = 0.0
        all_ratings_count = 0

        for ratings in self.movie_ratings.values():
            if ratings:
                all_ratings_sum += sum(ratings)
                all_ratings_count += len(ratings)
        
        self.global_mean = all_ratings_sum / all_ratings_count if all_ratings_count > 0 else 2.5

        C = 5.0 

        for movie_id, ratings in self.movie_ratings.items():
            n = len(ratings)
            if n > 0:
                bayesian_avg = (sum(ratings) + C * self.global_mean) / (n + C)
                self.movie_averages[movie_id] = bayesian_avg

        for user_id, user_obj in self.users.items():
            deviations = []
            for m_id, r in user_obj.ratings.items():
                m_avg = self.movie_averages.get(m_id, self.global_mean)
                deviations.append(r - m_avg)
            
            if deviations:
                self.user_deviations[user_id] = sum(deviations) / len(deviations)

    def rate(self, user, movie_id):
        m_avg = self.movie_averages.get(movie_id, self.global_mean)
        u_dev = self.user_deviations.get(user.id, 0.0)

        prediction = m_avg + u_dev

        return round(float(np.clip(prediction, 0.5, 5.0)) / 0.5) * 0.5

    def __str__(self):
        return 'System 155294 and 155877'

