import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class DecentralizedFeedRanking:
    def __init__(self, diversity_weight=0.3, novelty_weight=0.2, quality_weight=0.3,
                 popularity_weight=0.1, connection_weight=0.1):
        """
        Initialize the decentralized feed ranking algorithm with customizable weights.
        """
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.quality_weight = quality_weight
        self.popularity_weight = popularity_weight
        self.connection_weight = connection_weight

        # User data structures
        self.users = {}
        self.posts = {}
        self.user_interests = defaultdict(dict)
        self.user_engagement_history = defaultdict(list)
        self.content_categories = set()
        self.exposure_diversity = defaultdict(dict)
        self.social_graph = nx.Graph()

        # Content analysis
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # For visualization
        self.feed_history = defaultdict(list)
        self.diversity_metrics = defaultdict(list)

    def add_user(self, user_id, initial_interests=None, bias_factors=None):
        """Add a user to the system with initial interests and bias factors"""
        if initial_interests is None:
            initial_interests = {}

        if bias_factors is None:
            bias_factors = {
                'confirmation_bias': random.uniform(0.1, 0.9),
                'outgroup_bias': random.uniform(0.1, 0.9)
            }

        self.users[user_id] = {
            'interests': initial_interests,
            'bias_factors': bias_factors,
            'viewed_posts': set(),
            'interaction_count': defaultdict(int),
            'diversity_score': 0.5  # Default middle value
        }

        # Initialize user node in social graph
        self.social_graph.add_node(user_id)

        # Initialize exposure diversity tracking for all categories
        for category in self.content_categories:
            self.exposure_diversity[user_id][category] = 0.0

    def add_connection(self, user_id1, user_id2, strength=1.0):
        """Add a connection between two users with a given strength"""
        if user_id1 in self.users and user_id2 in self.users:
            self.social_graph.add_edge(user_id1, user_id2, weight=strength)

    def add_post(self, post_id, user_id, content, categories, quality_score=None,
                 harmful_content_score=0.0, misinformation_score=0.0,
                 polarization_score=0.0):
        """
        Add a post to the system.
        """
        # Update content categories
        for category in categories:
            self.content_categories.add(category)
            # Update exposure diversity tracking for all users
            for uid in self.users:
                if category not in self.exposure_diversity[uid]:
                    self.exposure_diversity[uid][category] = 0.0

        # Compute quality score if not provided (example implementation)
        if quality_score is None:
            content_length_factor = min(1.0, len(content) / 500)
            quality_score = content_length_factor * (1 - harmful_content_score) * (1 - misinformation_score)

        self.posts[post_id] = {
            'user_id': user_id,
            'content': content,
            'categories': categories,
            'quality_score': quality_score,
            'harmful_content_score': harmful_content_score,
            'misinformation_score': misinformation_score,
            'polarization_score': polarization_score,
            'engagement_count': 0,
            'creation_time': len(self.posts),  # Simple timestamp
            'vector': None  # To be computed
        }

    def compute_content_vectors(self):
        """Compute vector representations for all posts"""
        contents = [post['content'] for post in self.posts.values()]
        if not contents:
            return
        try:
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            for i, post_id in enumerate(self.posts.keys()):
                self.posts[post_id]['vector'] = tfidf_matrix[i]
        except ValueError:
            for post_id in self.posts:
                self.posts[post_id]['vector'] = None

    def calculate_content_similarity(self, post_id1, post_id2):
        """Calculate cosine similarity between two posts"""
        if self.posts[post_id1]['vector'] is None or self.posts[post_id2]['vector'] is None:
            return 0.0
        return cosine_similarity(
            self.posts[post_id1]['vector'],
            self.posts[post_id2]['vector']
        )[0][0]

    def record_user_engagement(self, user_id, post_id, engagement_type, engagement_strength):
        """
        Record user engagement with a post.
        """
        if user_id not in self.users or post_id not in self.posts:
            return

        self.user_engagement_history[user_id].append({
            'post_id': post_id,
            'type': engagement_type,
            'strength': engagement_strength,
            'time': len(self.user_engagement_history[user_id])
        })

        self.users[user_id]['viewed_posts'].add(post_id)
        self.posts[post_id]['engagement_count'] += engagement_strength

        # Update user interests based on post categories
        for category in self.posts[post_id]['categories']:
            if category not in self.user_interests[user_id]:
                self.user_interests[user_id][category] = 0.0
            self.user_interests[user_id][category] += engagement_strength * 0.1
            self.user_interests[user_id][category] = min(1.0, self.user_interests[user_id][category])

        # Update interaction count with post creator
        creator_id = self.posts[post_id]['user_id']
        self.users[user_id]['interaction_count'][creator_id] += 1

        # Update exposure diversity
        for category in self.posts[post_id]['categories']:
            self.exposure_diversity[user_id][category] += 1

    def calculate_user_content_affinity(self, user_id, post_id):
        """Calculate how much a post aligns with user interests"""
        if user_id not in self.users or post_id not in self.posts:
            return 0.0
        post_categories = self.posts[post_id]['categories']
        if not post_categories:
            return 0.0
        total_affinity = sum(self.user_interests[user_id].get(cat, 0.0) for cat in post_categories)
        return total_affinity / len(post_categories)

    def calculate_content_diversity_score(self, user_id, post_id):
        """
        Calculate how diverse a post is compared to user's recent engagement.
        """
        if user_id not in self.users or post_id not in self.posts:
            return 0.5

        recent_engagements = self.user_engagement_history[user_id][-10:]
        if not recent_engagements:
            return 0.5

        post_categories = set(self.posts[post_id]['categories'])
        category_exposure = defaultdict(int)
        total_categories_seen = 0

        for engagement in recent_engagements:
            engaged_post_id = engagement['post_id']
            if engaged_post_id in self.posts:
                for cat in self.posts[engaged_post_id]['categories']:
                    category_exposure[cat] += 1
                    total_categories_seen += 1

        if total_categories_seen == 0 or not post_categories:
            return 0.5

        total_exposure = sum(category_exposure.get(cat, 0) for cat in post_categories)
        avg_exposure = total_exposure / len(post_categories)
        normalized_diversity = 1.0 - (avg_exposure / total_categories_seen)
        return normalized_diversity

    def calculate_novelty_score(self, user_id, post_id):
        """Calculate how novel a post is based on creation time and user's history"""
        if user_id not in self.users or post_id not in self.posts:
            return 0.5

        max_time = max(post['creation_time'] for post in self.posts.values()) if self.posts else 0
        time_novelty = self.posts[post_id]['creation_time'] / max_time if max_time > 0 else 0.5

        post_categories = set(self.posts[post_id]['categories'])
        if not post_categories:
            category_novelty = 0.5
        else:
            category_exposure = [self.exposure_diversity[user_id].get(cat, 0) for cat in post_categories]
            avg_exposure = sum(category_exposure) / len(category_exposure)
            max_exposure = max(self.exposure_diversity[user_id].values()) if self.exposure_diversity[user_id] else 1
            category_novelty = 1.0 - (avg_exposure / max_exposure) if max_exposure > 0 else 0.5

        return 0.4 * time_novelty + 0.6 * category_novelty

    def calculate_social_connection_score(self, user_id, post_id):
        """Calculate social connection strength between user and post creator"""
        if user_id not in self.users or post_id not in self.posts:
            return 0.0
        post_creator = self.posts[post_id]['user_id']
        if user_id == post_creator:
            return 1.0
        if self.social_graph.has_edge(user_id, post_creator):
            return self.social_graph.edges[user_id, post_creator]['weight']
        interaction_count = self.users[user_id]['interaction_count'].get(post_creator, 0)
        return min(1.0, interaction_count / 10)

    def calculate_content_quality_score(self, post_id):
        """Return the quality score of a post with penalties for harmful content"""
        if post_id not in self.posts:
            return 0.5
        post = self.posts[post_id]
        base_quality = post['quality_score']
        harmful_penalty = post['harmful_content_score'] * 0.7
        misinfo_penalty = post['misinformation_score'] * 0.5
        polarization_penalty = post['polarization_score'] * 0.3
        total_penalty = min(0.9, harmful_penalty + misinfo_penalty + polarization_penalty)
        return max(0.1, base_quality * (1 - total_penalty))

    def rank_feed_for_user(self, user_id, candidate_post_ids=None, limit=10):
        """
        Rank posts for a user's feed.
        """
        if user_id not in self.users:
            return []
        if candidate_post_ids is None:
            viewed = self.users[user_id]['viewed_posts']
            candidate_post_ids = [pid for pid in self.posts if pid not in viewed]
        if not candidate_post_ids:
            return []

        self.compute_content_vectors()

        post_scores = []
        for pid in candidate_post_ids:
            affinity = self.calculate_user_content_affinity(user_id, pid)
            diversity = self.calculate_content_diversity_score(user_id, pid)
            novelty = self.calculate_novelty_score(user_id, pid)
            quality = self.calculate_content_quality_score(pid)
            connection = self.calculate_social_connection_score(user_id, pid)

            combined = (affinity * (1 - self.diversity_weight) +
                        diversity * self.diversity_weight +
                        novelty * self.novelty_weight +
                        quality * self.quality_weight +
                        connection * self.connection_weight)

            confirmation_bias = self.users[user_id]['bias_factors']['confirmation_bias']
            bias_adjusted = combined * (1 - confirmation_bias * 0.5) + affinity * (confirmation_bias * 0.5)
            post_scores.append((pid, bias_adjusted, {
                'content_affinity': affinity,
                'diversity_score': diversity,
                'novelty_score': novelty,
                'quality_score': quality,
                'connection_score': connection,
                'combined_score': combined,
                'bias_adjusted_score': bias_adjusted
            }))

        sorted_posts = sorted(post_scores, key=lambda x: x[1], reverse=True)
        detailed = {pid: scores for pid, _, scores in post_scores}
        self.feed_history[user_id].append({
            'timestamp': len(self.feed_history[user_id]),
            'posts': [(pid, detailed[pid]) for pid, _, _ in sorted_posts[:limit]]
        })

        avg_diversity = (sum(scores['diversity_score'] for _, _, scores in post_scores) /
                          len(post_scores)) if post_scores else 0
        self.diversity_metrics[user_id].append(avg_diversity)

        return [pid for pid, _, _ in sorted_posts[:limit]]

    def visualize_user_feed(self, user_id, feed_posts=None):
        """
        Visualize the factors affecting a user's feed ranking.
        """
        if user_id not in self.users:
            print(f"User {user_id} not found")
            return
        if feed_posts is None:
            feed_posts = self.rank_feed_for_user(user_id)
        if not feed_posts:
            print("No posts in feed to visualize")
            return
        if not self.feed_history[user_id]:
            print("No feed history available")
            return

        latest_feed = self.feed_history[user_id][-1]
        feed_data = {pid: scores for pid, scores in latest_feed['posts'] if pid in feed_posts}
        if not feed_data:
            print("No scoring data available for posts")
            return

        metrics = ['content_affinity', 'diversity_score', 'novelty_score',
                   'quality_score', 'connection_score', 'bias_adjusted_score']
        data = []
        for pid, scores in feed_data.items():
            row = {'post_id': pid}
            for m in metrics:
                row[m] = scores[m]
            data.append(row)
        df = pd.DataFrame(data)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        heat_df = df.set_index('post_id').drop('bias_adjusted_score', axis=1)
        colors = ['#E2EEFF', '#99BBFF', '#6689CC', '#335599', '#112266']
        cmap = LinearSegmentedColormap.from_list('blue_gradient', colors)
        sns.heatmap(heat_df, cmap=cmap, annot=True, fmt='.2f', cbar_kws={'label': 'Score'})
        plt.title(f'Feed Ranking Factors for User {user_id}')
        plt.ylabel('Posts')

        plt.subplot(2, 1, 2)
        final = df.sort_values('bias_adjusted_score', ascending=False)
        sns.barplot(x='post_id', y='bias_adjusted_score', data=final, palette='Blues_d')
        plt.title('Final Ranking Scores')
        plt.xlabel('Post ID')
        plt.ylabel('Final Score')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def visualize_diversity_over_time(self, user_id):
        """Visualize how content diversity changes over time for a user"""
        if user_id not in self.diversity_metrics or not self.diversity_metrics[user_id]:
            print(f"No diversity metrics available for User {user_id}")
            return
        metrics = self.diversity_metrics[user_id]
        timestamps = list(range(len(metrics)))
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, metrics, marker='o', linestyle='-', color='#335599')
        plt.title(f'Content Diversity Over Time for User {user_id}')
        plt.xlabel('Feed Updates')
        plt.ylabel('Average Diversity Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Diversity Threshold')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_category_exposure(self, user_id):
        """Visualize the distribution of content categories a user is exposed to"""
        if user_id not in self.exposure_diversity or not self.exposure_diversity[user_id]:
            print(f"No category exposure data available for User {user_id}")
            return
        exposure = self.exposure_diversity[user_id]
        cats = list(exposure.keys())
        vals = list(exposure.values())
        sorted_indices = np.argsort(vals)[::-1]
        sorted_cats = [cats[i] for i in sorted_indices]
        sorted_vals = [vals[i] for i in sorted_indices]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_cats, sorted_vals, color='#335599')
        max_val = max(vals) if vals else 0
        threshold = 0.7 * max_val
        for i, value in enumerate(sorted_vals):
            if value > threshold:
                bars[i].set_color('#CC3333')
        plt.title(f'Category Exposure Distribution for User {user_id}')
        plt.xlabel('Content Categories')
        plt.ylabel('Exposure Count')
        plt.xticks(rotation=45, ha='right')
        if max_val > 0:
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Echo Chamber Threshold')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def simulate_user_interactions(self, user_id, num_interactions=20, engagement_bias=0.7):
        """
        Simulate user interactions with feed content based on their interests and biases.
        """
        if user_id not in self.users:
            print(f"User {user_id} not found")
            return
        for _ in range(num_interactions):
            feed_posts = self.rank_feed_for_user(user_id, limit=10)
            if not feed_posts:
                break
            for pid in feed_posts[:5]:
                post_cats = set(self.posts[pid]['categories'])
                alignment = sum(self.user_interests[user_id].get(cat, 0) for cat in post_cats)
                alignment = alignment / len(post_cats) if post_cats else 0
                engagement_prob = alignment * engagement_bias + random.uniform(0, 1) * (1 - engagement_bias)
                if random.random() < engagement_prob:
                    strength = random.uniform(0.3, 1.0)
                    self.record_user_engagement(user_id, pid, 'like', strength)

    def run_feed_diversity_experiment(self, num_users=5, num_posts=50, num_categories=8,
                                      simulation_cycles=5, with_diversity=True):
        """
        Run an experiment to demonstrate the impact of diversity-focused feed ranking.
        """
        original_diversity_weight = self.diversity_weight
        self.diversity_weight = 0.4 if with_diversity else 0.05

        categories = [f"category_{i + 1}" for i in range(num_categories)]

        for i in range(num_users):
            user_id = f"user_{i + 1}"
            interests = {}
            for cat in categories:
                if random.random() < 0.3:
                    interests[cat] = random.uniform(0.3, 0.8)
            self.add_user(user_id, interests, {
                'confirmation_bias': random.uniform(0.3, 0.9),
                'outgroup_bias': random.uniform(0.3, 0.9)
            })

        for i in range(num_users):
            for j in range(i + 1, num_users):
                if random.random() < 0.3:
                    self.add_connection(f"user_{i + 1}", f"user_{j + 1}", random.uniform(0.3, 1.0))

        for i in range(num_posts):
            post_id = f"post_{i + 1}"
            user_id = f"user_{random.randint(1, num_users)}"
            num_cats = random.randint(1, 3)
            post_cats = random.sample(categories, num_cats)
            content = f"This is post {i + 1} about {', '.join(post_cats)}"
            quality_score = random.uniform(0.3, 1.0)
            harmful_score = random.uniform(0, 0.3)
            misinfo_score = random.uniform(0, 0.2)
            polarization_score = random.uniform(0, 0.4)
            self.add_post(post_id, user_id, content, post_cats,
                          quality_score, harmful_score, misinfo_score, polarization_score)

        diversity_trends = {uid: [] for uid in self.users}
        category_exposure_counts = {uid: defaultdict(int) for uid in self.users}

        for cycle in range(simulation_cycles):
            for uid in self.users:
                feed = self.rank_feed_for_user(uid)
                if feed and self.feed_history[uid]:
                    latest_feed = self.feed_history[uid][-1]
                    feed_data = {pid: scores for pid, scores in latest_feed['posts']}
                    avg_diversity = (sum(scores['diversity_score'] for scores in feed_data.values()) /
                                     len(feed_data)) if feed_data else 0
                    diversity_trends[uid].append(avg_diversity)
                    for pid in feed:
                        if pid in self.posts:
                            for cat in self.posts[pid]['categories']:
                                category_exposure_counts[uid][cat] += 1
                self.simulate_user_interactions(uid, num_interactions=5)

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        for uid, trends in diversity_trends.items():
            plt.plot(trends, marker='o', label=uid)
        plt.title(f"Feed Diversity Trends ({'With' if with_diversity else 'Without'} Diversity Focus)")
        plt.xlabel("Simulation Cycle")
        plt.ylabel("Average Diversity Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2)
        example_user = list(self.users.keys())[0]
        cats = list(category_exposure_counts[example_user].keys())
        counts = list(category_exposure_counts[example_user].values())
        sorted_indices = np.argsort(counts)[::-1]
        sorted_cats = [cats[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        plt.bar(sorted_cats, sorted_counts, color='#335599')
        plt.title(f"Category Exposure for {example_user}")
        plt.xlabel("Categories")
        plt.ylabel("Exposure Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        self.diversity_weight = original_diversity_weight

        return {
            'average_diversity': {uid: sum(trends) / len(trends) if trends else 0
                                  for uid, trends in diversity_trends.items()},
            'category_concentration': {uid: self._calculate_category_concentration(counts)
                                       for uid, counts in category_exposure_counts.items()}
        }

    def _calculate_category_concentration(self, category_counts):
        """Calculate Gini coefficient for category exposure"""
        values = list(category_counts.values())
        if not values or sum(values) == 0:
            return 0.0
        values.sort()
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / (np.sum(values) * n)) / n if n > 0 else 0.0

    def compare_ranking_strategies(self, user_id, strategies):
        """
        Compare different ranking strategies for the same user.
        """
        if user_id not in self.users:
            print(f"User {user_id} not found")
            return

        original_weights = {
            'diversity_weight': self.diversity_weight,
            'novelty_weight': self.novelty_weight,
            'quality_weight': self.quality_weight,
            'popularity_weight': self.popularity_weight,
            'connection_weight': self.connection_weight
        }

        strategy_feeds = {}
        strategy_metrics = {}

        for strat_name, weights in strategies.items():
            for param, value in weights.items():
                if hasattr(self, param):
                    setattr(self, param, value)
            feed = self.rank_feed_for_user(user_id)
            if feed and self.feed_history[user_id]:
                latest_feed = self.feed_history[user_id][-1]
                feed_data = {pid: scores for pid, scores in latest_feed['posts']}
                strategy_feeds[strat_name] = feed
                strategy_metrics[strat_name] = {
                    'diversity': np.mean([scores['diversity_score'] for scores in feed_data.values()]),
                    'novelty': np.mean([scores['novelty_score'] for scores in feed_data.values()]),
                    'quality': np.mean([scores['quality_score'] for scores in feed_data.values()]),
                    'affinity': np.mean([scores['content_affinity'] for scores in feed_data.values()]),
                    'connection': np.mean([scores['connection_score'] for scores in feed_data.values()])
                }
        for param, value in original_weights.items():
            setattr(self, param, value)
        if strategy_metrics:
            self._visualize_strategy_comparison(strategy_metrics)
        return strategy_feeds, strategy_metrics

    def _visualize_strategy_comparison(self, strategy_metrics):
        """Visualize comparison between different ranking strategies"""
        metrics = ['diversity', 'novelty', 'quality', 'affinity', 'connection']
        strategies = list(strategy_metrics.keys())
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        for strat, met_dict in strategy_metrics.items():
            values = [met_dict[m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=strat)
            ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Ranking Strategy Comparison')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def _eigenvector_centrality_disconnected(self, G, weight='weight'):
        """
        Helper method: Compute eigenvector centrality for each connected (or strongly connected)
        component in G and merge the results.
        """
        centrality = {}
        if G.is_directed():
            components = list(nx.strongly_connected_components(G))
        else:
            components = list(nx.connected_components(G))
        for comp in components:
            subG = G.subgraph(comp)
            if len(subG) == 1:
                # For a single-node component, assign zero centrality
                node = next(iter(subG.nodes()))
                centrality[node] = 0.0
            else:
                cent = nx.eigenvector_centrality_numpy(subG, weight=weight)
                centrality.update(cent)
        return centrality

    def analyze_social_influence(self):
        """
        Analyze how social connections influence content ranking.
        """
        if not self.social_graph.edges():
            print("No social connections to analyze")
            return

        # Create weighted directed graph of influence
        influence_graph = nx.DiGraph()
        for uid in self.users:
            influence_graph.add_node(uid)

        # Count posts from each creator in all feeds for each user
        for uid in self.users:
            all_feeds = self.feed_history.get(uid, [])
            if not all_feeds:
                continue
            creator_counts = defaultdict(int)
            total_posts = 0
            for feed in all_feeds:
                for pid, _ in feed.get('posts', []):
                    if pid in self.posts:
                        creator = self.posts[pid]['user_id']
                        creator_counts[creator] += 1
                        total_posts += 1
            if total_posts == 0:
                continue

            # For each creator (excluding self), add an edge if influence is significant
            for creator, count in creator_counts.items():
                if creator != uid:
                    influence_score = count / total_posts
                    if influence_score > 0.05:
                        influence_graph.add_edge(creator, uid, weight=influence_score)

        # Visualize the influence graph
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(influence_graph, seed=42)
        nx.draw_networkx_nodes(influence_graph, pos, node_size=500, node_color='lightblue', alpha=0.8)
        edge_weights = [influence_graph[u][v]['weight'] * 5 for u, v in influence_graph.edges()]
        nx.draw_networkx_edges(influence_graph, pos, width=edge_weights, edge_color='gray',
                               alpha=0.6, arrowsize=15, arrowstyle='->')
        nx.draw_networkx_labels(influence_graph, pos, font_size=10)
        plt.title('Social Influence Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Use the helper method to handle disconnected graphs
        influence_centrality = self._eigenvector_centrality_disconnected(influence_graph, weight='weight')
        influence_betweenness = nx.betweenness_centrality(influence_graph, weight='weight')
        return {
            'influence_centrality': influence_centrality,
            'influence_betweenness': influence_betweenness
        }

    def export_network_data(self, filename_prefix):
        """Export network data for external analysis."""
        social_edges = []
        for u, v, data in self.social_graph.edges(data=True):
            social_edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 1.0)
            })
        social_df = pd.DataFrame(social_edges)
        social_df.to_csv(f"{filename_prefix}_social_network.csv", index=False)

        interest_data = []
        for uid, interests in self.user_interests.items():
            for cat, level in interests.items():
                interest_data.append({
                    'user_id': uid,
                    'category': cat,
                    'interest_level': level
                })
        interest_df = pd.DataFrame(interest_data)
        interest_df.to_csv(f"{filename_prefix}_user_interests.csv", index=False)

        exposure_data = []
        for uid, cats in self.exposure_diversity.items():
            for cat, count in cats.items():
                exposure_data.append({
                    'user_id': uid,
                    'category': cat,
                    'exposure_count': count
                })
        exposure_df = pd.DataFrame(exposure_data)
        exposure_df.to_csv(f"{filename_prefix}_exposure_diversity.csv", index=False)

        print(f"Network data exported with prefix: {filename_prefix}")

# Example usage
if __name__ == "__main__":
    feed_ranker = DecentralizedFeedRanking(
        diversity_weight=0.3,
        novelty_weight=0.2,
        quality_weight=0.3,
        popularity_weight=0.1,
        connection_weight=0.1
    )

    users = ["user1", "user2", "user3", "user4", "user5"]
    categories = ["tech", "politics", "sports", "entertainment", "science"]

    feed_ranker.add_user("user1", {"tech": 0.8, "science": 0.7})
    feed_ranker.add_user("user2", {"politics": 0.9, "entertainment": 0.6})
    feed_ranker.add_user("user3", {"sports": 0.8, "tech": 0.4})
    feed_ranker.add_user("user4", {"entertainment": 0.9, "politics": 0.5})
    feed_ranker.add_user("user5", {"science": 0.8, "sports": 0.3})

    feed_ranker.add_connection("user1", "user3", 0.7)
    feed_ranker.add_connection("user1", "user5", 0.6)
    feed_ranker.add_connection("user2", "user4", 0.8)
    feed_ranker.add_connection("user3", "user5", 0.5)
    feed_ranker.add_connection("user4", "user5", 0.4)

    for i in range(30):
        user_id = random.choice(users)
        post_cats = random.sample(categories, random.randint(1, 2))
        content = f"Post {i + 1} about {' and '.join(post_cats)}"
        feed_ranker.add_post(
            f"post{i + 1}",
            user_id,
            content,
            post_cats,
            quality_score=random.uniform(0.5, 1.0),
            harmful_content_score=random.uniform(0, 0.2),
            misinformation_score=random.uniform(0, 0.1),
            polarization_score=random.uniform(0, 0.3)
        )

    print("Running feed diversity experiment...")
    results = feed_ranker.run_feed_diversity_experiment(
        num_users=5,
        num_posts=50,
        num_categories=8,
        simulation_cycles=10,
        with_diversity=True
    )

    print("\nComparing ranking strategies...")
    strategies = {
        "diversity_focused": {"diversity_weight": 0.6, "novelty_weight": 0.2},
        "quality_focused": {"quality_weight": 0.6, "diversity_weight": 0.1},
        "social_focused": {"connection_weight": 0.6, "diversity_weight": 0.1},
        "balanced": {"diversity_weight": 0.3, "novelty_weight": 0.2,
                     "quality_weight": 0.3, "connection_weight": 0.2}
    }
    feeds, metrics = feed_ranker.compare_ranking_strategies("user1", strategies)

    print("\nAnalyzing social influence network...")
    centrality_metrics = feed_ranker.analyze_social_influence()
    print("Centrality metrics:", centrality_metrics)

    feed_ranker.export_network_data("feed_experiment")
    print("\nExperiment complete!")
