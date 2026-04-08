from RatingSystem import RatingSystem
from collections import defaultdict


class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()

        self.item_users_ranks = defaultdict(dict)
        self.sim_cache = {}

        # global avg
        global_sum = 0
        global_count = 0
        for user_obj in self.users.values():
            for rating in user_obj.ratings.values():
                global_sum += rating
                global_count += 1
        self.global_mean = global_sum / global_count if global_count > 0 else 2.5

        # base biases
        lmbda_movie = 7.0
        lmbda_user = 5.0

        movie_bias_sum = defaultdict(float)
        movie_bias_count = defaultdict(int)
        for user_obj in self.users.values():
            for m_id, rating in user_obj.ratings.items():
                movie_bias_sum[m_id] += (rating - self.global_mean)
                movie_bias_count[m_id] += 1

        self.b_i = {m: movie_bias_sum[m] / (movie_bias_count[m] + lmbda_movie) for m in movie_bias_sum}

        user_bias_sum = defaultdict(float)
        for u_id, user_obj in self.users.items():
            for m_id, rating in user_obj.ratings.items():
                user_bias_sum[u_id] += (rating - self.global_mean - self.b_i.get(m_id, 0.0))

        self.b_u = {u: user_bias_sum[u] / (len(self.users[u].ratings) + lmbda_user) for u in user_bias_sum}

        # spearman ranking
        for user_id, user_obj in self.users.items():
            ratings = user_obj.ratings
            if not ratings:
                continue

            sorted_items = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
            current_rank = 1.0
            i = 0
            n_items = len(sorted_items)

            while i < n_items:
                j = i
                while j < n_items and sorted_items[j][1] == sorted_items[i][1]:
                    j += 1
                n_ties = j - i
                avg_rank = current_rank + (n_ties - 1) / 2.0

                for k in range(i, j):
                    m_id = sorted_items[k][0]
                    self.item_users_ranks[m_id][user_id] = avg_rank

                current_rank += n_ties
                i = j

        # parameters
        self.shrinkage_lambda = 15.0
        self.k_neighbors = 30

    def _get_similarity(self, m1, m2):
        pair = (min(m1, m2), max(m1, m2))

        if pair in self.sim_cache:
            return self.sim_cache[pair]

        ranks1 = self.item_users_ranks.get(m1, {})
        ranks2 = self.item_users_ranks.get(m2, {})

        common_users = ranks1.keys() & ranks2.keys()
        n = len(common_users)

        if n < 3:
            self.sim_cache[pair] = 0.0
            return 0.0

        sum_d_sq = sum((ranks1[u] - ranks2[u]) ** 2 for u in common_users)
        rho = 1.0 - (6.0 * sum_d_sq) / (n * (n * n - 1.0))

        shrinkage_factor = n / (n + self.shrinkage_lambda)
        shrunk_rho = rho * shrinkage_factor

        self.sim_cache[pair] = shrunk_rho
        return shrunk_rho

    def rate(self, user, movie):
        target_movie = movie
        u_id = user.id

        # baseline: avg + user bias + movie bias
        bu = self.b_u.get(u_id, 0.0)
        bi_target = self.b_i.get(target_movie, 0.0)
        baseline_target = self.global_mean + bu + bi_target

        # fallback
        if target_movie not in self.item_users_ranks:
            return min(max(baseline_target, 1.0), 5.0)

        similarities = []
        for rated_movie_id, user_rating in user.ratings.items():
            if rated_movie_id not in self.item_users_ranks:
                continue

            sim = self._get_similarity(target_movie, rated_movie_id)
            if sim > 0:
                similarities.append((sim, user_rating, rated_movie_id))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similarities = similarities[:self.k_neighbors]

        numerator = 0.0
        denominator = 0.0

        for sim, user_rating, rated_movie_id in top_similarities:
            # diviation
            bi_rated = self.b_i.get(rated_movie_id, 0.0)
            baseline_rated = self.global_mean + bu + bi_rated

            rating_diff = user_rating - baseline_rated

            numerator += sim * rating_diff
            denominator += sim

        if denominator > 0:
            prediction = baseline_target + (numerator / denominator)
            return min(max(prediction, 1.0), 5.0)
        else:
            return min(max(baseline_target, 1.0), 5.0)

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 155877'