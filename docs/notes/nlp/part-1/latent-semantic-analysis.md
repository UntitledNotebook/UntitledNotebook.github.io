# Latent semantic analysis

## The term–document matrix

Let $\mathcal V=\{w_1,\ldots,w_m\}$ be the vocabulary and $\mathcal D=\{d_1,\ldots,d_n\}$ be a collection of nonempty documents. The **term–document (TD) matrix** is $W_{\mathrm{td}}\in\mathbb R^{m\times n}$, with entries

$$
(W_{\mathrm{td}})_{ij}=c_{ij},
\qquad
c_{ij}=\text{the number of occurrences of }w_i\text{ in }d_j.
$$

Row $i$ represents word $w_i$ across the documents. Column $j$ represents document $d_j$ as a bag of words.

Three properties of raw counts motivate LSA.

1. **Direct overlap.** The dot product between two word rows measures direct document overlap. If no document contains both words, their dot product is zero. For example, let $e_1$ be `cat paw` and $e_2$ be `dog paw`, with each listed word occurring once. The rows for *cat* and *dog* are $(1,0)$ and $(0,1)$, so their dot product is zero, even though the shared word *paw* links their contexts.
2. **Sparsity.** Each word has one coordinate per document, and most words appear in only a small fraction of documents. The resulting word vectors are high-dimensional and mostly zero.
3. **Unequal weights.** Long documents tend to have larger counts, and common words such as *the* can dominate a decomposition of the raw matrix, even though their presence does little to distinguish documents.

LSA uses TF-IDF to adjust the weights and truncated SVD to represent words and documents in a shared low-dimensional space.

## Singular value decomposition

For $A\in\mathbb R^{m\times n}$, a **singular value decomposition (SVD)** is

$$
A=U\Sigma V^\top,
\qquad
U\in\mathbb R^{m\times m},\quad
\Sigma\in\mathbb R^{m\times n},\quad
V\in\mathbb R^{n\times n}.
$$

The matrices $U$ and $V$ are orthogonal. The only possibly nonzero entries of $\Sigma$ are on its main diagonal. They are the singular values $\sigma_1\ge\cdots\ge\sigma_p\ge 0$, where $p=\min(m,n)$.

Let $r=\operatorname{rank}(A)$ and let $u_\ell$ and $v_\ell$ be columns $\ell$ of $U$ and $V$. Exactly $r$ singular values are positive, so

$$
A=\sum_{\ell=1}^{r}\sigma_\ell u_\ell v_\ell^\top
 =U_r\Sigma_rV_r^\top.
$$

The factors have shapes $m\times r$, $r\times r$, and $r\times n$, respectively. Every term in the sum is an $m\times n$ rank-one matrix.

When $A$ is a TD matrix, each row of $U$ corresponds to a word. The rows of the full, square $U$ are orthonormal, so using them as word vectors would give a cosine similarity of zero for every pair of distinct words.

## Best rank-$k$ approximation

For $1\le k\le r$, let $U_k$ and $V_k$ contain the first $k$ columns of $U$ and $V$, and let $\Sigma_k=\operatorname{diag}(\sigma_1,\ldots,\sigma_k)$. The **truncated SVD** is

$$
A_k=U_k\Sigma_kV_k^\top
   =\sum_{\ell=1}^k\sigma_\ell u_\ell v_\ell^\top.
$$

**Theorem (Eckart–Young).** The truncated SVD achieves the minimum squared Frobenius error among matrices of rank at most $k$.

$$
\min_{\substack{B\in\mathbb R^{m\times n}\\\operatorname{rank}(B)\le k}}
\|A-B\|_F^2
=\|A-A_k\|_F^2
=\sum_{\ell=k+1}^{r}\sigma_\ell^2,
\qquad
\|X\|_F^2=\sum_{i,j}X_{ij}^2.
$$

**Proof.** Let $P$ be the orthogonal projector onto the column space of any candidate $B$ of rank at most $k$. Since $(I-P)B=0$, the orthogonal decomposition of each column gives

$$
\|A-B\|_F^2
=\|P(A-B)\|_F^2+\|(I-P)A\|_F^2
\ge \|A\|_F^2-\|PA\|_F^2.
$$

The orthogonality of the right singular vectors gives

$$
\|PA\|_F^2=\sum_{\ell=1}^r \sigma_\ell^2 a_\ell,
\qquad
a_\ell=\|Pu_\ell\|_2^2.
$$

Each $a_\ell$ lies in $[0,1]$, and $\sum_{\ell=1}^r a_\ell\le\operatorname{tr}(P)\le k$. Allocating this weight to the $k$ largest squared singular values gives the upper bound $\|PA\|_F^2\le\sum_{\ell=1}^k\sigma_\ell^2$. Thus $\|A-B\|_F^2\ge\sum_{\ell=k+1}^r\sigma_\ell^2$. The choice $B=A_k$ attains this bound because the omitted rank-one terms are orthogonal in the Frobenius inner product. $\square$

## Latent word and document vectors

Let $z_i=\bigl((U_k)_{i,:}\bigr)^\top$ be the $k$-dimensional word vector for $w_i$, and let $h_j=\Sigma_k\bigl((V_k)_{j,:}\bigr)^\top$ be the document vector for $d_j$. The reconstruction is

$$
(A_k)_{ij}=z_i^\top h_j\approx A_{ij}.
$$

If nonnegative association scores are desired, one can take $\max\{z_i^\top h_j,0\}$. The Frobenius optimality theorem applies to the reconstruction before this clipping.

The columns of $U_k$ remain orthonormal. Its rows can have nonzero inner products, so word vectors can reflect shared patterns across documents. The identity $AV_k=U_k\Sigma_k$ shows how the original word rows are projected onto the retained directions.

**Example (two documents).** Return to the documents $e_1=$ `cat paw` and $e_2=$ `dog paw` introduced under direct overlap. With rows ordered as *cat*, *dog*, and *paw*, their TD matrix is

$$
W_{\mathrm{td}}=
\begin{pmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{pmatrix}.
$$

Apply SVD to this raw count matrix to see the effect of truncation alone. Its leading right singular vector is $v_1=(1,1)^\top/\sqrt{2}$. Projecting the *cat* and *dog* rows onto this direction gives both the coordinate $1/\sqrt{2}$. The corresponding rows of $U_1$ are equal as well, since the projection coordinates are divided by the same singular value. The shared occurrence of *paw* in $e_1$ and $e_2$ makes this common direction stronger than the direction that separates the documents.

## TF-IDF normalization

Let $L_j$ be the number of tokens in document $d_j$. The **term frequency (TF)** is the fraction of tokens in $d_j$ that are instances of $w_i$.

$$
\mathrm{TF}_{ij}=\frac{c_{ij}}{L_j}.
$$

The **document frequency** is $\mathrm{df}_i=|\{j:c_{ij}>0\}|$. The smoothed **inverse document frequency (IDF)** gives greater weight to words that occur in fewer documents.

$$
\mathrm{IDF}_i=1+\ln\!\left(\frac{n+1}{\mathrm{df}_i+1}\right).
$$

Each document containing $w_i$ contributes one to $\mathrm{df}_i$, regardless of the number of occurrences. A word that appears in every document has $\mathrm{IDF}_i=1$.

The TF-IDF weighted TD matrix $A$ has entries

$$
A_{ij}=\mathrm{TF}_{ij}\,\mathrm{IDF}_i
      =\frac{c_{ij}\,\mathrm{IDF}_i}{L_j}.
$$

Each occurrence of $w_i$ contributes $\mathrm{IDF}_i$ to the numerator. Dividing by $L_j$ gives its average contribution per token in the document.

**Example (seven documents).** Consider seven documents $d_1,\ldots,d_7$, represented as bags of words over the vocabulary $\{\textit{cat},\textit{dog},\textit{the}\}$. Their complete word counts are:

| Word | $d_1$ | $d_2$ | $d_3$ | $d_4$ | $d_5$ | $d_6$ | $d_7$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| *cat* | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| *dog* | 0 | 8 | 0 | 1 | 0 | 0 | 0 |
| *the* | 20 | 13 | 18 | 22 | 15 | 4 | 20 |
| **Document length** | **21** | **22** | **18** | **23** | **15** | **4** | **20** |

For instance, $d_2$ contains one *cat*, eight *dog* tokens, and thirteen *the* tokens, for a total of 22 tokens. Both *cat* and *dog* occur in two documents, whereas *the* occurs in all seven.

The three word rows form the TD matrix; the document-length row records the column sums. The table gives $n=7$, $L_2=1+8+13=22$, $\mathrm{df}_{\textit{cat}}=\mathrm{df}_{\textit{dog}}=2$, and $\mathrm{df}_{\textit{the}}=7$. Substituting these counts into the TF and smoothed IDF formulas gives:

| Word | Count in $d_2$ | Document frequency | TF | IDF | TF-IDF |
|---|---:|---:|---:|---:|---:|
| *cat* | 1 | 2 | $1/22$ | $1+\ln(8/3)\approx 1.981$ | $0.090$ |
| *dog* | 8 | 2 | $8/22$ | $1+\ln(8/3)\approx 1.981$ | $0.720$ |
| *the* | 13 | 7 | $13/22$ | $1$ | $0.591$ |

The weight of *dog* in $d_2$ exceeds that of *the* because *dog* occurs in fewer documents.

## Full approach

1. Build the count matrix $W_{\mathrm{td}}$ from the vocabulary and document collection.
2. Compute TF-IDF to obtain the weighted matrix $A$.
3. Compute the leading $k$ singular directions of $A$ and form $A_k=U_k\Sigma_kV_k^\top$.
4. Use the rows of $U_k$ as word embeddings and the columns of $\Sigma_kV_k^\top$ as document embeddings. Word similarity can be measured by cosine similarity between nonzero word vectors. The dot product between a word vector and a document vector gives the corresponding reconstructed TF-IDF weight.
