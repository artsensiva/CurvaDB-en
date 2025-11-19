"""Main CurveDB class for Level 2.

Integrates curve fitting, FPCA, and Hilbert indexing for semantic search.
"""

import os
import pickle
from typing import List, Tuple, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import lmdb

from .curve_fitter import CurveFitter, Curve
from .fpca import FunctionalPCA
from .distances import DISTANCE_FUNCTIONS


class CurveDB:
    """Curve-based semantic search database (MVP).

    Architecture:
    1. Text → Sentence embeddings → Curves (B-splines)
    2. Curves → FPCA → Low-dimensional features
    3. FPCA features → Hilbert curve → 1D index
    4. Search: Hilbert range query + Curve distance reranking

    Args:
        db_path: Directory to store database files
        embedding_model: Name of sentence-transformers model
        n_fpca_components: Number of FPCA components
        hilbert_order: Order of Hilbert curve
        curve_degree: Degree of B-spline curves
        distance_metric: Distance metric for reranking ('frechet', 'dtw', 'l2', 'cosine')
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_fpca_components: int = 10,
        hilbert_order: int = 8,
        curve_degree: int = 3,
        distance_metric: str = 'frechet',
        n_sample_points: int = 50,
        curve_dim: int = 8  # Dimension for curve space (must be < 11 for scipy's splprep)
    ):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.n_fpca_components = n_fpca_components
        self.hilbert_order = hilbert_order
        self.curve_degree = curve_degree
        self.distance_metric = distance_metric
        self.n_sample_points = n_sample_points
        self.curve_dim = curve_dim

        os.makedirs(db_path, exist_ok=True)

        # Initialize components
        print(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # PCA to reduce embedding dimension for curve fitting (scipy splprep requires dim < 11)
        self.embedding_pca = PCA(n_components=curve_dim)
        self.embedding_pca_fitted = False

        self.curve_fitter = CurveFitter(degree=curve_degree)
        self.fpca = FunctionalPCA(
            n_components=n_fpca_components,
            n_sample_points=n_sample_points
        )

        # Storage
        self.lmdb_env = None
        self.curves: List[Curve] = []
        self.fpca_scores: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.fitted = False
        self.all_chunk_embeddings: List[np.ndarray] = []  # For fitting PCA

        # Try to load existing database
        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing database if available."""
        fpca_path = os.path.join(self.db_path, 'fpca.pkl')
        curves_path = os.path.join(self.db_path, 'curves.pkl')
        texts_path = os.path.join(self.db_path, 'texts.pkl')
        scores_path = os.path.join(self.db_path, 'fpca_scores.npy')
        embedding_pca_path = os.path.join(self.db_path, 'embedding_pca.pkl')

        if all(os.path.exists(p) for p in [fpca_path, curves_path, texts_path, scores_path]):
            print("Loading existing database...")
            self.fpca = FunctionalPCA.load(fpca_path)

            with open(curves_path, 'rb') as f:
                self.curves = pickle.load(f)

            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)

            self.fpca_scores = np.load(scores_path)

            # Load embedding PCA if available
            if os.path.exists(embedding_pca_path):
                with open(embedding_pca_path, 'rb') as f:
                    self.embedding_pca = pickle.load(f)
                    self.embedding_pca_fitted = True

            self.fitted = True

            print(f"Loaded {len(self.curves)} curves from database")

    def _text_to_token_embeddings(self, text: str) -> np.ndarray:
        """Convert text to sequence of token embeddings.

        Uses transformer's native token-level embeddings instead of
        chunk embeddings for better granularity with short texts.

        Args:
            text: Input text

        Returns:
            Token embeddings, shape (n_tokens, embedding_dim)
        """
        # Get token embeddings from SentenceTransformer
        # This returns per-token embeddings instead of pooled sentence embedding
        embeddings = self.encoder.encode(
            text,
            output_value='token_embeddings',
            convert_to_numpy=False,  # Returns torch tensor
            show_progress_bar=False
        )

        # Convert to numpy
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def add_documents(self, texts: List[str], show_progress: bool = True):
        """Add documents to the database.

        Args:
            texts: List of text documents
            show_progress: Whether to show progress bar
        """
        print(f"Adding {len(texts)} documents...")

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # Pass 1: Collect all chunk embeddings for PCA fitting
        print("Pass 1: Collecting chunk embeddings...")
        all_embeddings = []
        doc_embeddings = []

        iterator = tqdm(texts, desc="Collecting embeddings") if (show_progress and has_tqdm) else texts
        for text in iterator:
            embeddings = self._text_to_token_embeddings(text)
            doc_embeddings.append(embeddings)
            all_embeddings.append(embeddings)

        # Fit embedding PCA on all chunks
        if not self.embedding_pca_fitted:
            print("Fitting embedding PCA...")
            stacked = np.vstack(all_embeddings)
            self.embedding_pca.fit(stacked)
            self.embedding_pca_fitted = True
            print(f"PCA fitted: {self.embedding_dim}D -> {self.curve_dim}D")

        # Pass 2: Apply PCA and fit curves
        print("Pass 2: Fitting curves...")
        new_curves = []
        valid_texts = []

        iterator = tqdm(zip(texts, doc_embeddings), total=len(texts), desc="Converting to curves") if (show_progress and has_tqdm) else zip(texts, doc_embeddings)
        for text, embeddings in iterator:
            # Apply PCA to reduce dimension
            reduced_embeddings = self.embedding_pca.transform(embeddings)

            # Fit curve in reduced space
            curve = self.curve_fitter.tokens_to_curve(reduced_embeddings)

            if curve is not None:
                new_curves.append(curve)
                valid_texts.append(text)
            else:
                print(f"Warning: Failed to fit curve for text: {text[:50]}...")

        print(f"Successfully created {len(new_curves)} curves from {len(texts)} documents")

        # Add to storage
        self.curves.extend(new_curves)
        self.texts.extend(valid_texts)

        # Mark as not fitted (need to refit FPCA)
        self.fitted = False

    def fit(self):
        """Fit FPCA on all curves and build index."""
        if len(self.curves) < self.n_fpca_components:
            raise ValueError(
                f"Need at least {self.n_fpca_components} curves to fit FPCA, "
                f"got {len(self.curves)}"
            )

        print(f"Fitting FPCA on {len(self.curves)} curves...")

        # Fit FPCA
        self.fpca.fit(self.curves)

        # Transform all curves to FPCA space
        print("Transforming curves to FPCA space...")
        self.fpca_scores = self.fpca.transform(self.curves)

        print(f"FPCA scores shape: {self.fpca_scores.shape}")
        print(f"Explained variance ratio: {self.fpca.explained_variance_ratio()[:5]}")

        self.fitted = True

    def search(
        self,
        query: str,
        k: int = 10,
        rerank_factor: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            rerank_factor: How many candidates to retrieve before reranking (k * rerank_factor)

        Returns:
            List of (text, distance) tuples, sorted by distance (lower is better)
        """
        if not self.fitted:
            raise RuntimeError("Database must be fitted before search")

        # Convert query to curve
        query_embeddings = self._text_to_token_embeddings(query)
        # Apply PCA to reduce dimension (same as in add_documents)
        reduced_embeddings = self.embedding_pca.transform(query_embeddings)
        query_curve = self.curve_fitter.tokens_to_curve(reduced_embeddings)

        # Flag for whether we can do curve-based reranking
        can_rerank = query_curve is not None

        if can_rerank:
            # Get FPCA score for query
            query_score = self.fpca.transform([query_curve])[0]
        else:
            # Fallback: use mean of reduced embeddings as pseudo-FPCA score
            # This allows search to still work even if curve fitting fails
            mean_embedding = reduced_embeddings.mean(axis=0)
            # Pad or truncate to match FPCA dimensions
            if len(mean_embedding) < self.n_fpca_components:
                query_score = np.pad(mean_embedding, (0, self.n_fpca_components - len(mean_embedding)))
            else:
                query_score = mean_embedding[:self.n_fpca_components]

        # Find candidates using Hilbert curve
        n_candidates = min(k * rerank_factor, len(self.curves))

        # Compute distances in FPCA space
        fpca_distances = np.linalg.norm(
            self.fpca_scores - query_score,
            axis=1
        )

        # Get top candidates
        candidate_indices = np.argpartition(fpca_distances, n_candidates)[:n_candidates]
        candidate_indices = candidate_indices[np.argsort(fpca_distances[candidate_indices])]

        # Rerank using curve distance (if possible)
        if can_rerank:
            distance_fn = DISTANCE_FUNCTIONS[self.distance_metric]

            reranked = []
            for idx in candidate_indices:
                try:
                    dist = distance_fn(
                        query_curve,
                        self.curves[idx],
                        n_samples=self.n_sample_points
                    )
                    reranked.append((idx, dist))
                except Exception as e:
                    # Fallback to FPCA distance for this candidate
                    reranked.append((idx, fpca_distances[idx]))

            # Sort by distance and take top k
            reranked.sort(key=lambda x: x[1])
            results = [(self.texts[idx], dist) for idx, dist in reranked[:k]]
        else:
            # No curve reranking - use FPCA distances directly
            results = [(self.texts[idx], fpca_distances[idx]) for idx in candidate_indices[:k]]

        return results

    def save(self):
        """Save database to disk."""
        if not self.fitted:
            raise RuntimeError("Database must be fitted before saving")

        print(f"Saving database to {self.db_path}...")

        # Save FPCA model
        fpca_path = os.path.join(self.db_path, 'fpca.pkl')
        self.fpca.save(fpca_path)

        # Save curves
        curves_path = os.path.join(self.db_path, 'curves.pkl')
        with open(curves_path, 'wb') as f:
            pickle.dump(self.curves, f)

        # Save texts
        texts_path = os.path.join(self.db_path, 'texts.pkl')
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)

        # Save FPCA scores
        scores_path = os.path.join(self.db_path, 'fpca_scores.npy')
        np.save(scores_path, self.fpca_scores)

        # Save embedding PCA
        embedding_pca_path = os.path.join(self.db_path, 'embedding_pca.pkl')
        with open(embedding_pca_path, 'wb') as f:
            pickle.dump(self.embedding_pca, f)

        # Save metadata
        metadata = {
            'embedding_model': self.embedding_model_name,
            'n_fpca_components': self.n_fpca_components,
            'hilbert_order': self.hilbert_order,
            'curve_degree': self.curve_degree,
            'distance_metric': self.distance_metric,
            'n_sample_points': self.n_sample_points,
            'n_documents': len(self.texts)
        }
        metadata_path = os.path.join(self.db_path, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print("Database saved successfully")

    def stats(self) -> Dict:
        """Get database statistics."""
        stats = {
            'n_documents': len(self.texts),
            'n_curves': len(self.curves),
            'embedding_dim': self.embedding_dim,
            'n_fpca_components': self.n_fpca_components,
            'fitted': self.fitted,
            'distance_metric': self.distance_metric
        }

        if self.fitted:
            stats['fpca_scores_shape'] = self.fpca_scores.shape
            stats['explained_variance_ratio'] = self.fpca.explained_variance_ratio().tolist()

        return stats

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return f"CurveDB(documents={len(self.texts)}, curves={len(self.curves)}, {status})"
