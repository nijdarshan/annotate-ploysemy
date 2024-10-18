-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    word_list_toggle BOOLEAN DEFAULT FALSE
);

-- Words table
CREATE TABLE words (
    id SERIAL PRIMARY KEY,
    word_name VARCHAR(100) UNIQUE NOT NULL,
    directory_path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(id),
    sentence_id INTEGER NOT NULL,
    dimension_1 FLOAT NOT NULL,
    dimension_2 FLOAT NOT NULL,
    UNIQUE(word_id, sentence_id)
);

-- Sentences table
CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(id),
    sentence_index INTEGER NOT NULL,
    sentence_text TEXT NOT NULL,
    UNIQUE(word_id, sentence_index)
);

-- Cluster labels table
CREATE TABLE cluster_labels (
    id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(id),
    sentence_id INTEGER NOT NULL,
    cluster_label INTEGER NOT NULL,
    UNIQUE(word_id, sentence_id)
);

-- Annotations table
CREATE TABLE annotations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    word_id INTEGER REFERENCES words(id),
    sentence_id INTEGER NOT NULL,
    cluster_number INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, word_id, sentence_id)
);

-- Completion status table
CREATE TABLE completion_status (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    word_id INTEGER REFERENCES words(id),
    is_completed BOOLEAN DEFAULT FALSE,
    cluster1_count INTEGER DEFAULT 0,
    cluster2_count INTEGER DEFAULT 0,
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, word_id)
);
