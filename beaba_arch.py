"""beaba_arch.py

BEABA: A Character-to-Visual-Word Holistic Tokenizer and Algebraic Graph-State Memory.

Architecture overview:
1. FRONT-END: Holistic Word-Form Encoder (Cognitive VWFA / LCD model):
   - Characters enter sequentially.
   - Computes open-bigram binding (SERIOL lag weighting) + terminal-relative anchors
     (first/last letters weighted more, mirroring human transposed-letter reading) +
     fractional-power position encoding (FPE Gaussian uncertainty kernel).
   - Produces holistic word hypervectors w in C^1800 (unit-norm R^3600).

2. STATE MEMORY: Holographic Graph State G in C^1800 (3600 real dims):
   - Updates at word boundaries: G_t = e^{-i theta} * G_{t-1} + w_t
   - In this space:
     (a) SUBTRACTION: G(s) - w('marrom') rotated by e^{+i theta} == G(s without 'marrom')
     (b) QUERY / UNBINDING: query(G, 'raposa') finds offset k* and recovers continuation:
         'pulou o cachorro marrom'
     (c) GRAPH STRUCTURE: acts as a succession graph via phase shifts or 60x60 matrix edge memory.

3. BACK-END: Char Generative Decoder / Lexical Cleanup Memory.
"""
import math, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cpu'
torch.set_num_threads(8)
D = 1800  # 1800 complex = 3600 real dimensions

class HolisticWordFormEncoder:
    """
    Computes a visual/cognitive holistic word-form hypervector w(word) in C^D.
    Implements:
    - Position-tolerant letter bag (Agrawal et al. 2020 eLife)
    - Ordered open-bigrams with lag decay (Dehaene LCD / Whitney SERIOL)
    - Exterior letter anchoring (Cox et al. 2011 terminal-relative / scRNN)
    - Position uncertainty via Fractional Power Encoding (FPE)
    """
    def __init__(self, D=D, seed=42, ell=1.2, max_gap=3, decay=0.6, w_edge=2.0):
        self.D = D
        self.max_gap = max_gap
        self.decay = decay
        self.w_edge = w_edge
        
        rs = np.random.RandomState(seed)
        # Position kernel frequencies: k(d) = exp(-d^2 / (2 ell^2))
        self.theta_b = rs.normal(0, 1.0 / ell, D).astype(np.float32)
        self.theta_e = rs.normal(0, 1.0 / ell, D).astype(np.float32)
        self.perm = rs.permutation(D)
        
        # Base letter phasor codebook
        self._letter_phases = {}
        for c in "abcdefghijklmnopqrstuvwxyzàáâãéêíóôõúç_<>":
            self._letter_phases[c] = rs.uniform(-np.pi, np.pi, D).astype(np.float32)
            
    def get_letter_phase(self, c):
        if c not in self._letter_phases:
            # Hash OOV chars
            h = int(abs(hash(c))) % (2**31)
            self._letter_phases[c] = np.random.RandomState(h).uniform(-np.pi, np.pi, self.D).astype(np.float32)
        return self._letter_phases[c]

    def encode_word(self, word):
        """Returns unit-norm phasor vector w in C^D (and its R^3600 representation)."""
        w = word.lower().strip()
        L = len(w)
        if L == 0:
            return torch.zeros(self.D, dtype=torch.complex64, device=DEV)
            
        acc = np.zeros(self.D, dtype=np.complex64)
        
        # 1. Letter positions with start- and end-anchoring
        for i, c in enumerate(w):
            ph = self.get_letter_phase(c)
            # Edge letters have higher weight (exterior anchoring)
            edge_mult = self.w_edge if (i == 0 or i == L - 1) else 1.0
            
            # Start-anchored position: phase + i * theta_b
            ph_b = ph + i * self.theta_b
            acc += edge_mult * np.exp(1j * ph_b)
            
            # End-anchored position: phase + (L - 1 - i) * theta_e
            ph_e = ph + (L - 1 - i) * self.theta_e
            acc += edge_mult * np.exp(1j * ph_e)
            
        # 2. Open-bigrams (pairs within window max_gap)
        for i in range(L):
            ph_i = self.get_letter_phase(w[i])
            for j in range(i + 1, min(L, i + 1 + self.max_gap)):
                ph_j = self.get_letter_phase(w[j])[self.perm]
                weight = self.decay ** (j - i - 1)
                acc += weight * np.exp(1j * (ph_i + ph_j))
                
        # 3. Boundary markers
        ph_start = self.get_letter_phase('<')
        ph_end = self.get_letter_phase('>')
        acc += self.w_edge * np.exp(1j * (ph_start + self.get_letter_phase(w[0])[self.perm]))
        acc += self.w_edge * np.exp(1j * (self.get_letter_phase(w[-1]) + ph_end[self.perm]))
        
        # Phase projection: project each component to unit circle (FHRR phasor)
        phases = np.angle(acc)
        z = np.exp(1j * phases).astype(np.complex64)
        tensor_z = torch.from_numpy(z).to(DEV)
        return tensor_z

    def to_real_3600(self, z):
        """Converts C^1800 phasor into unit-normalized R^3600 [Re, Im]."""
        real_vec = torch.cat([torch.real(z), torch.imag(z)], dim=-1)
        return F.normalize(real_vec, p=2, dim=-1)


class BeabaAlgebraicMemory:
    """
    State Memory in C^D (3600 real dimensions).
    Maintains G_t = e^{-i theta} * G_{t-1} + w_t.
    """
    def __init__(self, D=D, seed=123):
        self.D = D
        rs = np.random.RandomState(seed)
        theta = rs.uniform(-np.pi, np.pi, D).astype(np.float32)
        self.theta = torch.from_numpy(theta).to(DEV)
        self.word_encoder = HolisticWordFormEncoder(D=D, seed=seed)
        self.codebook = {} # word -> phasor z
        
    def rot(self, k):
        """Rotation factor exp(-i * k * theta)"""
        angle = -k * self.theta
        return torch.complex(torch.cos(angle), torch.sin(angle))
        
    def get_word_vector(self, word):
        if word not in self.codebook:
            self.codebook[word] = self.word_encoder.encode_word(word)
        return self.codebook[word]

    def encode_text(self, text):
        """
        Processes text word by word (or character stream with space boundaries).
        Updates state G incrementally.
        """
        words = re.findall(r"[a-zà-öø-ÿ]+", text.lower())
        T = len(words)
        G = torch.zeros(self.D, dtype=torch.complex64, device=DEV)
        for t, w in enumerate(words):
            offset = T - 1 - t
            w_vec = self.get_word_vector(w)
            G = G + w_vec * self.rot(offset)
        return G, words

    def subtract_word(self, G, word, is_last=True, offset=0):
        """
        Algebraic Subtraction:
        - If is_last=True: removes the most recent word and rotates back:
          G_prev = (G - w) * exp(+i * theta)
        - If hole deletion at arbitrary offset:
          G_hole = G - w * exp(-i * offset * theta)
        """
        w_vec = self.get_word_vector(word)
        if is_last:
            G_sub = G - w_vec
            inv_rot = torch.conj(self.rot(1))
            return G_sub * inv_rot
        else:
            return G - w_vec * self.rot(offset)

    def locate_word(self, G, word, max_search=64):
        """
        Unbinds query word and locates its offset k*:
        c = G * conj(w)
        score(k) = Re < c, rot(k) > / D
        """
        w_vec = self.get_word_vector(word)
        c = G * torch.conj(w_vec)
        
        k_tensor = torch.arange(max_search, device=DEV).unsqueeze(1)
        angles = -k_tensor.float() * self.theta.unsqueeze(0)
        rot_all = torch.complex(torch.cos(angles), torch.sin(angles))
        
        scores = torch.real(torch.sum(c.unsqueeze(0) * torch.conj(rot_all), dim=-1)) / self.D
        best_k = torch.argmax(scores).item()
        return best_k, scores

    def decode_word_at_offset(self, G, offset, candidates=None):
        """
        Cleans up the representation at offset k against candidates (or internal codebook).
        """
        if candidates is None:
            candidates = list(self.codebook.keys())
            
        target = G * torch.conj(self.rot(offset)) # [D]
        
        # Build candidate matrix
        cand_keys = list(candidates)
        cand_vecs = torch.stack([self.get_word_vector(w) for w in cand_keys]) # [V, D]
        
        scores = torch.real(torch.sum(cand_vecs * torch.conj(target.unsqueeze(0)), dim=-1)) / self.D
        best_idx = torch.argmax(scores).item()
        return cand_keys[best_idx], scores[best_idx].item()

    def query_continuation(self, G, query_word, candidates=None, max_search=64):
        """
        Query Q = query_word:
        1. Find offset k* where query_word was placed.
        2. Decode offsets k*-1 down to 0 to yield the continuation string.
        """
        k_star, _ = self.locate_word(G, query_word, max_search=max_search)
        continuation_words = []
        for off in range(k_star - 1, -1, -1):
            w_pred, _ = self.decode_word_at_offset(G, off, candidates=candidates)
            continuation_words.append(w_pred)
        return " ".join(continuation_words)


def demo():
    print("=" * 70)
    print("BEABA ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    memory = BeabaAlgebraicMemory(D=1800, seed=42)
    
    # 1. Holistic visual reading (Transposition Robustness)
    print("\n1. Visual Word-Form Encoding (Transposition Invariance):")
    v_clean = memory.word_encoder.encode_word("raposa")
    v_transposed = memory.word_encoder.encode_word("rpaosa") # interior transposition
    v_scrambled = memory.word_encoder.encode_word("rpoasa")  # interior scramble
    v_sub = memory.word_encoder.encode_word("ratosa")        # substitution
    v_unrelated = memory.word_encoder.encode_word("cachorro")
    
    def cos_sim(a, b):
        return (torch.real(torch.sum(a * torch.conj(b))) / memory.D).item()
        
    print(f"  Similarity('raposa', 'raposa'):    {cos_sim(v_clean, v_clean):.4f}")
    print(f"  Similarity('raposa', 'rpaosa'):    {cos_sim(v_clean, v_transposed):.4f} (Transposed letters / Cmabrigde)")
    print(f"  Similarity('raposa', 'rpoasa'):    {cos_sim(v_clean, v_scrambled):.4f} (Interior Scramble)")
    print(f"  Similarity('raposa', 'ratosa'):    {cos_sim(v_clean, v_sub):.4f} (Substitution)")
    print(f"  Similarity('raposa', 'cachorro'):  {cos_sim(v_clean, v_unrelated):.4f} (Unrelated / Quasi-orthogonal)")

    # 2. Encoding the phrase into 3600-dim algebraic state
    text = "a raposa pulou o cachorro marrom"
    print(f"\n2. Encoding phrase into 3600-dim state G:")
    print(f"   '{text}'")
    G, words = memory.encode_text(text)
    print(f"   State G created. Complex dim: {memory.D}, Real dim: {2 * memory.D}")

    # 3. Subtraction
    print("\n3. Algebraic Subtraction:")
    print("   G_sub = G - w('marrom')")
    G_sub = memory.subtract_word(G, "marrom", is_last=True)
    
    target_text = "a raposa pulou o cachorro"
    G_target, _ = memory.encode_text(target_text)
    
    diff = torch.norm(G_sub - G_target).item()
    sim = cos_sim(G_sub, G_target) / (torch.norm(G_sub).item() * torch.norm(G_target).item() / memory.D)
    print(f"   Target text: '{target_text}'")
    print(f"   Error || G_sub - G_target || = {diff:.6e}")
    print(f"   Cosine similarity            = {sim:.8f} (Exact!)")

    # 4. Query & Continuation
    query_word = "raposa"
    print(f"\n4. Query Q = '{query_word}':")
    continuation = memory.query_continuation(G, query_word, candidates=words)
    print(f"   Retrieved continuation: '{continuation}'")
    print(f"   Expected continuation:  'pulou o cachorro marrom'")
    print(f"   Match: {continuation == 'pulou o cachorro marrom'}")

if __name__ == "__main__":
    demo()
