import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging

logging.basicConfig(filename='app.log', level=logging.ERROR)

class Database:
    def __init__(self, config):
        self.config = config

    @contextmanager
    def get_db_cursor(self):
        conn = psycopg2.connect(**self.config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                yield cursor
                conn.commit()
        finally:
            conn.close()

    def get_word_list(self, user_id):
        try:
            with self.get_db_cursor() as cur:
                cur.execute("""
                    SELECT w.*, cs.is_completed, cs.cluster1_count, cs.cluster2_count
                    FROM words w
                    LEFT JOIN completion_status cs ON w.id = cs.word_id AND cs.user_id = %s
                    ORDER BY w.word_name
                """, (user_id,))
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Error fetching word list for user {user_id}: {e}")
            raise

    def get_word_data(self, word_id):
        with self.get_db_cursor() as cur:
            cur.execute("""
                SELECT sentence_id, dimension_1, dimension_2
                FROM embeddings
                WHERE word_id = %s
            """, (word_id,))
            embeddings = cur.fetchall()

            cur.execute("""
                SELECT sentence_index, sentence_text
                FROM sentences
                WHERE word_id = %s
            """, (word_id,))
            sentences = cur.fetchall()

            cur.execute("""
                SELECT sentence_id, cluster_label
                FROM cluster_labels
                WHERE word_id = %s
            """, (word_id,))
            cluster_labels = cur.fetchall()

            return embeddings, sentences, cluster_labels

    def save_annotation(self, user_id, word_id, sentence_id, cluster_number):
        try:
            with self.get_db_cursor() as cur:
                cur.execute("""
                    INSERT INTO annotations (user_id, word_id, sentence_id, cluster_number)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, word_id, sentence_id)
                    DO UPDATE SET cluster_number = EXCLUDED.cluster_number
                """, (user_id, word_id, sentence_id, cluster_number))

                cur.execute("""
                    INSERT INTO completion_status (user_id, word_id, cluster1_count, cluster2_count)
                    VALUES (%s, %s, 
                        (SELECT COUNT(*) FROM annotations WHERE user_id = %s AND word_id = %s AND cluster_number = 1),
                        (SELECT COUNT(*) FROM annotations WHERE user_id = %s AND word_id = %s AND cluster_number = 2))
                    ON CONFLICT (user_id, word_id)
                    DO UPDATE SET 
                        cluster1_count = (SELECT COUNT(*) FROM annotations WHERE user_id = %s AND word_id = %s AND cluster_number = 1),
                        cluster2_count = (SELECT COUNT(*) FROM annotations WHERE user_id = %s AND word_id = %s AND cluster_number = 2),
                        is_completed = CASE 
                            WHEN (SELECT COUNT(*) FROM annotations WHERE user_id = %s AND word_id = %s AND cluster_number = 1) >= 40
                                AND (SELECT COUNT(*) FROM annotations WHERE user_id = %s AND word_id = %s AND cluster_number = 2) >= 40
                            THEN TRUE
                            ELSE FALSE
                        END
                """, (user_id, word_id, user_id, word_id, user_id, word_id, user_id, word_id, user_id, word_id, user_id, word_id, user_id, word_id))
        except Exception as e:
            logging.error(f"Error saving annotation for user {user_id}: {e}")
            raise

    def toggle_word_list(self, user_id):
        try:
            with self.get_db_cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET word_list_toggle = NOT word_list_toggle
                    WHERE id = %s
                    RETURNING word_list_toggle
                """, (user_id,))
                return cur.fetchone()['word_list_toggle']
        except Exception as e:
            logging.error(f"Error toggling word list for user {user_id}: {e}")
            raise
