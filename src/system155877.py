import csv
from RatingSystem import RatingSystem

# baseline + genres
class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()

        # global average
        total_sum = 0
        total_count = 0
        for user_obj in self.users.values():
            for rating in user_obj.ratings.values():
                total_sum += rating
                total_count += 1

        self.global_mean = total_sum / total_count if total_count > 0 else 2.5

        # loading genres via separate method
        self._load_genres()

        # biases
        lambda_movie = 7.0
        lambda_user = 5.0

        # diff from global avg (movies)
        movie_sums = {}
        movie_counts = {}
        for user_obj in self.users.values():
            for m_id, rating in user_obj.ratings.items():
                movie_sums[m_id] = movie_sums.get(m_id, 0) + (rating - self.global_mean)
                movie_counts[m_id] = movie_counts.get(m_id, 0) + 1

        self.b_i = {m: s / (movie_counts[m] + lambda_movie) for m, s in movie_sums.items()}

        # diff from global avg (users)
        user_sums = {}
        for u_id, user_obj in self.users.items():
            u_sum = 0
            for m_id, rating in user_obj.ratings.items():
                u_sum += (rating - self.global_mean - self.b_i.get(m_id, 0.0))
            user_sums[u_id] = u_sum

        self.b_u = {u: s / (len(self.users[u].ratings) + lambda_user) for u, s in user_sums.items()}

        # include genres
        self.user_genre_profiles = {}
        for u_id, user_obj in self.users.items():
            g_sums = {}
            g_counts = {}
            for m_id, rating in user_obj.ratings.items():
                for genre in self.movie_genres.get(m_id, []):
                    g_sums[genre] = g_sums.get(genre, 0) + rating
                    g_counts[genre] = g_counts.get(genre, 0) + 1

            u_personal_mean = self.global_mean + self.b_u.get(u_id, 0.0)

            profile = {}
            for genre in g_sums:
                # bayes smoothing
                profile[genre] = (g_sums[genre] + 3 * u_personal_mean) / (g_counts[genre] + 3)

            self.user_genre_profiles[u_id] = profile

    def _load_genres(self):
        self.movie_genres = {}
        try:
            with open('../data/movie.csv', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    m_id = int(row[0])
                    genres = row[2].split('|') if len(row) > 2 else []
                    self.movie_genres[m_id] = genres
        except FileNotFoundError:
            print("File not found: '../data/movie.csv'")

    def rate(self, user, movie):
        u_id = user.id
        m_id = movie

        # baseline estimate
        bu = self.b_u.get(u_id, 0.0)
        bi = self.b_i.get(m_id, 0.0)
        baseline_pred = self.global_mean + bu + bi

        # estimate based on genres
        movie_genres = self.movie_genres.get(m_id, [])
        user_profile = self.user_genre_profiles.get(u_id, {})

        genre_pred = baseline_pred  # default value

        if movie_genres and user_profile:
            known_genre_scores = [user_profile[g] for g in movie_genres if g in user_profile]
            if known_genre_scores:
                genre_pred = sum(known_genre_scores) / len(known_genre_scores)

        # how much baseline is to be trusted
        alpha = 0.75

        final_prediction = alpha * baseline_pred + (1 - alpha) * genre_pred

        return min(max(final_prediction, 1.0), 5.0)

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania.
        """
        return 'System created by 155877'