// Vite + React + Pyodide Single Page App for Chapter 8: Linear Algebra for Machine Learning
// File: src/App.jsx

import { useEffect, useRef, useState } from "react";

const SECTIONS = [
  { id: "intro", title: "1. Introduction to Linear Algebra for ML" },
  { id: "vectors", title: "2. Vectors and Matrices" },
  { id: "operations", title: "3. Operations with Vectors and Matrices" },
  { id: "equations", title: "4. Solving Linear Equations" },
  { id: "eigen", title: "5. Eigenvalues and Eigenvectors" },
  { id: "feature-eng", title: "6. Practice: Feature Engineering with Real Data" },
];

// Reusable Code Runner Component
function CodeRunner({ title, description, initialCode }) {
  const [code, setCode] = useState(initialCode.trim());
  const [isEditing, setIsEditing] = useState(false);
  const [output, setOutput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const outputRef = useRef(null);

  // Ensure pyodide is loaded globally once
  async function ensurePyodide() {
    if (!window.pyodideLoading) {
      window.pyodideLoading = (async () => {
        const pyodide = await window.loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/" });
        // Preload common packages
        await pyodide.loadPackage(["numpy", "micropip", "matplotlib", "scikit-learn"]);      
        // Use non-interactive backend for inline rendering
        await pyodide.runPythonAsync(`
import micropip
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
        `);
        window.pyodide = pyodide;
        return pyodide;
      })();
    }
    return window.pyodideLoading;
  }

  const handleRun = async () => {
    try {
      setIsRunning(true);
      const pyodide = await ensurePyodide();

      // Redirect stdout
      let captured = "";
      pyodide.setStdout({
        batched: (s) => {
          captured += s + "\n";
        },
      });
      pyodide.setStderr({
        batched: (s) => {
          captured += s + "\n";
        },
      });

      // Provide helper to show plots inline using SVG
      const wrappedCode = `
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
__inline_svgs = []

def __capture_current_figure():
    buf = BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)
    svg_data = buf.read().decode('utf-8')
    __inline_svgs.append(svg_data)
    plt.close()

# Decode and run user code from base64 to avoid escaping issues
import base64
user_code = base64.b64decode("${btoa(unescape(encodeURIComponent(code)))}").decode('utf-8')
exec(user_code)
`;

      await pyodide.runPythonAsync(wrappedCode);

      // Grab any inline SVGs we captured by inspecting the globals
      const svgs = (await pyodide.runPythonAsync("__inline_svgs")).toJs();

      // Build HTML: text output + inline SVGs
      let html = "";
      if (captured.trim()) {
        // Justified left, line-by-line
        html += captured
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .split("\n")
          .map((line) => `<div style="text-align:left; white-space:pre;">${line}</div>`)
          .join("");
      }
      if (Array.isArray(svgs) && svgs.length > 0) {
        html += svgs
          .map(
            (svg) => `<div style="margin-top:0.75rem; border:1px solid #334155; border-radius:0.5rem; overflow:hidden;">${svg}</div>`
          )
          .join("");
      }
      setOutput(html || "(no output)");
      setIsRunning(false);
      setTimeout(() => {
        if (outputRef.current) {
          outputRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 50);
    } catch (err) {
      setIsRunning(false);
      setOutput(`<div style="color:#fecaca; text-align:left; white-space:pre-wrap;">${err}</div>`);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      alert("Code copied to clipboard");
    } catch (e) {
      alert("Failed to copy code");
    }
  };

  const toggleEdit = () => {
    setIsEditing((prev) => !prev);
  };

  return (
    <div className="mb-10 rounded-2xl bg-slate-900/90 border border-slate-700 text-slate-100 shadow-xl">
      <div className="px-5 pt-4 pb-2 border-b border-slate-700 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">{title}</h2>
          {description && <p className="text-lg text-slate-200 mt-1">{description}</p>}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleRun}
            className="px-3 py-1.5 text-sm rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold disabled:opacity-60"
            disabled={isRunning}
          >
            {isRunning ? "Running..." : "Run"}
          </button>
          <button
            onClick={toggleEdit}
            className="px-3 py-1.5 text-sm rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold"
          >
            {isEditing ? "View" : "Edit"}
          </button>
          <button
            onClick={handleCopy}
            className="px-3 py-1.5 text-sm rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold"
          >
            Copy
          </button>
        </div>
      </div>

      <div className="gap-0">
        {/* Code Block */}
        <div className="p-4 border-b border-slate-800 bg-slate-950/80">
          <div className="text-xs uppercase tracking-wide text-slate-400 mb-2">Python Code</div>
          <pre className="text-sm leading-relaxed text-slate-100 bg-slate-900 rounded-xl p-3 overflow-auto border border-slate-700">
            {isEditing ? (
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full h-64 bg-transparent outline-none resize-vertical font-mono text-sm text-slate-100"
              />
            ) : (
              <code>{code}</code>
            )}
          </pre>
        </div>

        {/* Output Block */}
        <div className="p-4 bg-slate-900/95" ref={outputRef}>
          <div className="text-xs uppercase tracking-wide text-slate-400 mb-2">Standard Output</div>
          <div
            className="min-h-[4rem] text-sm bg-slate-900 rounded-xl p-3 border border-slate-700 overflow-auto text-left"
            style={{ whiteSpace: "normal" }}
            dangerouslySetInnerHTML={{ __html: output }}
          />
        </div>
      </div>
    </div>
  );
}

function Section({ id, title, children }) {
  return (
    <section id={id} className="scroll-mt-24 mb-12">
      <h2 className="text-2xl font-bold text-slate-50 mb-3">{title}</h2>
      {children}
    </section>
  );
}

function App() {
  const [active, setActive] = useState("intro");

  useEffect(() => {
    const handler = () => {
      const offsets = SECTIONS.map((s) => {
        const el = document.getElementById(s.id);
        if (!el) return { id: s.id, top: Infinity };
        return { id: s.id, top: Math.abs(el.getBoundingClientRect().top) };
      });
      offsets.sort((a, b) => a.top - b.top);
      if (offsets[0]) setActive(offsets[0].id);
    };
    window.addEventListener("scroll", handler);
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex">
      {/* Sidebar */}
      <aside className="hidden lg:flex lg:flex-col w-72 border-r border-slate-800 bg-slate-950/95 sticky top-0 h-screen px-5 py-6 gap-6">
        <div>
          <h1 className="text-2xl font-semibold text-slate-50">Chapter 8</h1>
          <p className="text-sm text-slate-300">Linear Algebra for Machine Learning</p>
        </div>
        <nav className="flex-1 overflow-auto pr-1">
          <div className="text-sm uppercase tracking-wide text-slate-400 mb-2">Sections</div>
          <ul className="space-y-1">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => scrollToSection(s.id)}
                  className={`w-full text-left px-3 py-2 rounded-xl text-sm transition-colors border ${
                    active === s.id
                      ? "bg-sky-500 text-slate-950 border-sky-400"
                      : "bg-slate-900/80 text-slate-200 border-slate-700 hover:bg-slate-800"
                  }`}
                >
                  {s.title}
                </button>
              </li>
            ))}
          </ul>
        </nav>
        <div className="text-xs text-slate-500 leading-relaxed">
          <p className="font-semibold text-slate-300 mb-1">Study Tip</p>
          <p>Run each block, then switch to edit mode and tweak one line to see how the output changes.</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 max-w-4xl mx-auto px-4 sm:px-6 lg:px-10 py-8">
        <header className="mb-8 border-b border-slate-800 pb-5">
          <p className="text-xs font-mono tracking-[0.2em] text-sky-300 mb-2 uppercase">
            Python Data Analysis with Machine Learning · Vol. 2
          </p>
          <h1 className="text-4xl sm:text-4xl font-bold text-slate-50 mb-2">
            Chapter 8 · Linear Algebra for Machine Learning
          </h1>
          <p className="text-slate-300 max-w-2xl text-xl lg:text-base">
            Explore vectors, matrices, linear systems, and eigenvalues/eigenvectors with hands-on NumPy and inline plotting using Pyodide.
          </p>
        </header>

        {/* 1. Intro */}
        <Section id="intro" title="1. Introduction to Linear Algebra for ML">
          <p className="text-slate-200 text-lg text-left  leading-relaxed mb-4">
            Linear algebra provides the language for representing datasets, model parameters, and transformations in machine learning. In this
            chapter, we treat data as matrices, features as vectors, and algorithms as linear transformations acting on these objects.
          </p>
          <ul className="list-disc list-inside  text-orange-200 text-left text-lg mb-4 space-y-1">
            <li>Datasets as matrices: rows = samples, columns = features.</li>
            <li>Model parameters as vectors or matrices (e.g., weight vectors in regression).</li>
            <li>Algorithms like PCA and neural networks are sequences of matrix operations.</li>
          </ul>

          <CodeRunner
            title="Intro Example: Dataset and Parameter Shapes"
            description="Represent a toy dataset and parameter vector, then compute predictions using matrix multiplication."
            initialCode={`
import numpy as np

# 4 samples, 3 features
X = np.array([
    [1.2, 3.5, 5.1],
    [2.1, 0.3, 3.2],
    [0.5, 1.1, 0.9],
    [4.2, 2.2, 1.4],
])

# parameter vector (3 features)
w = np.array([0.5, -0.2, 0.1])

print("X shape: ", X.shape)
print("w shape: ", w.shape)

# predictions = Xw
preds = X @ w
print("Predictions:")
for i, p in enumerate(preds, start=1):
    print(f"  Sample {i}: {p:.3f}")
`}
          />

          <CodeRunner
            title="Intro Example: Visualizing a Linear Model in 2D"
            description="Plot a simple 2D dataset and a fitted regression line to see linear algebra in action."
            initialCode={`
import numpy as np
import matplotlib.pyplot as plt

# Simple 2D dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 2.0, 2.4, 3.8, 4.2])

# Construct design matrix with bias term
X = np.column_stack([np.ones_like(x), x])

# Normal equation: theta = (X^T X)^{-1} X^T y
theta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Coefficients (bias, slope): ", theta)

# Predicted line
x_line = np.linspace(0, 6, 100)
X_line = np.column_stack([np.ones_like(x_line), x_line])
y_line = X_line @ theta

plt.figure()
plt.scatter(x, y, label="data")
plt.plot(x_line, y_line, label="fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simple Linear Regression via Normal Equation")
plt.legend()

# capture the figure
__capture_current_figure()
`}
          />
        </Section>

        {/* 2. Vectors & Matrices */}
        <Section id="vectors" title="2. Vectors and Matrices">
          <p className="text-slate-200 text-lg text-left  leading-relaxed mb-4">
            Vectors represent points or directions in space; matrices represent linear transformations or stacked collections of vectors.
            Understanding their shapes and basic properties is essential for reasoning about machine learning models.
          </p>

          <CodeRunner
            title="Vectors and Matrices: Creation and Basic Inspection"
            description="Create vectors and matrices, then inspect shapes and ranks."
            initialCode={`
import numpy as np

v = np.array([2, 4, 6])
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Vector v: ", v)
print("Matrix M: ", M)
print("v shape: ", v.shape)
print("M shape: ", M.shape)

rank = np.linalg.matrix_rank(M)
print("Rank of M:", rank)
`}
          />

          <CodeRunner
            title="Determinant and Inverse"
            description="Compute the determinant and inverse of a matrix; observe what happens when the matrix is singular."
            initialCode={`
import numpy as np

A = np.array([[2, 1],
              [5, 3]])

print("A = ", A)

detA = np.linalg.det(A)
print("det(A) = ", detA)

if abs(detA) < 1e-9:
    print("Matrix is (numerically) singular; no inverse.")
else:
    invA = np.linalg.inv(A)
    print("A^{-1} =", invA)
    # check A * A^{-1} ≈ I
    I = A @ invA
    print("A @ A^{-1} =", I)
`}
          />
        </Section>

        {/* 3. Operations with Vectors & Matrices */}
        <Section id="operations" title="3. Operations with Vectors and Matrices">
          <p className="text-slate-200 text-lg text-left  leading-relaxed mb-4">
            Operations like dot products, norms, and matrix multiplication form the backbone of optimization and model evaluation in
            machine learning. Cosine similarity, projection, and gradient updates are all expressed using these primitives.
          </p>

          <CodeRunner
            title="Dot Product, Norm, and Cosine Similarity"
            description="Compute similarity between feature vectors using the dot product and cosine similarity."
            initialCode={`
import numpy as np

u = np.array([3, 4])
v = np.array([4, 3])

print("u =", u)
print("v =", v)

# dot product
uv_dot = np.dot(u, v)
print("u·v =", uv_dot)

# norms
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
print("||u|| =", norm_u)
print("||v|| =", norm_v)

# cosine similarity
cos_sim = uv_dot / (norm_u * norm_v)
print("cosine similarity =", cos_sim)
`}
          />

          <CodeRunner
            title="Matrix Multiplication in a Tiny Neural Network Layer"
            description="Use matrix multiplication to simulate a linear layer in a neural network."
            initialCode={`
import numpy as np

# 3 input features → 2 output units
W = np.array([[0.2, 0.8],
              [0.5, -0.1],
              [0.3, 0.4]])

b = np.array([0.1, -0.2])

# batch of 4 samples
X = np.array([
    [1.0, 0.5, 2.0],
    [0.5, 1.5, 1.0],
    [2.0, 1.0, 0.5],
    [0.1, 0.2, 0.3],
])

# linear transformation: XW + b
Z = X @ W + b
print("Linear layer output (no activation):", Z)
`}
          />
        </Section>

        {/* 4. Solving Linear Equations */}
        <Section id="equations" title="4. Solving Linear Equations">
          <p className="text-slate-200 text-lg text-left  leading-relaxed mb-4">
            Many estimation problems in ML reduce to solving linear systems. For small problems, we can solve Ax = b directly; for large ones,
            we use iterative methods but the underlying structure is still linear algebra.
          </p>

          <CodeRunner
            title="Solving a 2×2 System"
            description="Solve a small linear system Ax = b using numpy.linalg.solve."
            initialCode={`
import numpy as np

# System:
# 2x + 3y = 8
# 3x + 4y = 11

A = np.array([[2, 3],
              [3, 4]])

b = np.array([8, 11])

x = np.linalg.solve(A, b)
print("Solution x =", x)

# Verify Ax ≈ b
print("A @ x =", A @ x)
`}
          />

          <CodeRunner
            title="Overdetermined System and Least Squares"
            description="Solve an overdetermined system using the least-squares solution (as in linear regression)."
            initialCode={`
import numpy as np

# 4 equations, 2 unknowns (overdetermined)
A = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
])

b = np.array([2.1, 2.9, 3.7, 4.1])

# Least squares solution
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print("Least squares solution x =", x)
print("Residual sum of squares =", residuals)
`}
          />
        </Section>

        {/* 5. Eigenvalues & Eigenvectors */}
        <Section id="eigen" title="5. Eigenvalues and Eigenvectors">
          <p className="text-slate-200 text-lg text-left  leading-relaxed mb-4">
            Eigenvalues and eigenvectors reveal the principal directions of linear transformations. In ML, they underpin PCA, spectral
            clustering, and stability analysis of optimization algorithms.
          </p>

          <CodeRunner
            title="Eigenvalues and Eigenvectors of a 2×2 Matrix"
            description="Compute eigenvalues/eigenvectors and interpret them as directions scaled by the matrix."
            initialCode={`
import numpy as np

A = np.array([[4, -2],
              [1,  1]])

vals, vecs = np.linalg.eig(A)

print("A =", A)
print("Eigenvalues:", vals)
print("Eigenvectors (columns):", vecs)

# Verify A v ≈ λ v for the first eigenpair
lam = vals[0]
v = vecs[:, 0]
left = A @ v
right = lam * v
print("Check first eigenpair:")
print("A v =", left)
print("λ v =", right)
`}
          />

          <CodeRunner
            title="Toy PCA in 2D"
            description="Compute the principal component of a tiny 2D dataset and visualize the projection."
            initialCode={`
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
])

# Center the data
Xc = X - X.mean(axis=0)

# Covariance matrix
cov = np.cov(Xc, rowvar=False)

# Eigen-decomposition
vals, vecs = np.linalg.eig(cov)

# Sort by descending eigenvalue
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

pc1 = vecs[:, 0]

print("Eigenvalues:", vals)
print("First principal component:", pc1)

# Project data onto PC1
proj = Xc @ pc1

# Reconstruct back in 2D for plotting
X_proj = np.outer(proj, pc1)

plt.figure()
plt.scatter(Xc[:, 0], Xc[:, 1], label="original (centered)")
plt.scatter(X_proj[:, 0], X_proj[:, 1], label="projection on PC1")

# draw PC1 axis
origin = np.array([[0, 0]])
axis = np.vstack([origin, pc1 * 3])
plt.plot(axis[:, 0], axis[:, 1], label="PC1 axis")

plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.legend()
plt.title("PCA in 2D: Projection onto First Principal Component")
plt.xlabel("x1 (centered)")
plt.ylabel("x2 (centered)")

__capture_current_figure()
`}
          />
        </Section>

        {/* 6. Practice: Feature Engineering */}
        <Section id="feature-eng" title="6. Practice: Feature Engineering with Real Data">
          <p className="text-slate-200 text-lg text-left  leading-relaxed mb-4">
            Finally, we connect linear algebra operations directly with feature engineering on a real dataset. We will use scikit-learn's
            built-in breast cancer dataset, apply normalization, and run PCA to generate new features.
          </p>

          <CodeRunner
            title="Feature Scaling and Norms"
            description="Load the dataset, inspect shapes, and compute norms of feature vectors."
            initialCode={`
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load dataset
breast = load_breast_cancer()
X = breast.data

print("Data shape (samples, features):", X.shape)

# Take one sample and compute its L2 norm
x0 = X[0]
norm_x0 = np.linalg.norm(x0)
print("First sample L2 norm:", norm_x0)

# Normalize the first sample
x0_unit = x0 / norm_x0
print("First 5 entries of normalized sample:", x0_unit[:5])
`}
          />

          <CodeRunner
            title="PCA for Feature Engineering"
            description="Perform PCA to reduce the dataset to 2 dimensions and visualize the transformed features."
            initialCode={`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

breast = load_breast_cancer()
X = breast.data
y = breast.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratios:", pca.explained_variance_ratio_)

# Scatter plot of the first two principal components
plt.figure()
for label, marker, name in [(0, "o", "malignant"), (1, "^", "benign")]:
    mask = (y == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], marker=marker, label=name, alpha=0.6)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Breast Cancer Dataset")
plt.legend()

__capture_current_figure()
`}
          />

          <div className="mt-4 p-4 rounded-2xl bg-slate-900/80 border border-slate-700 text-sm text-slate-200">
            <h3 className="text-base font-semibold mb-2 text-slate-50">Study Tips for This Chapter</h3>
            <ul className="list-disc text-orange-200 list-inside text-left space-y-1">
              <li>For each code block, first run it as-is, then switch to <span className="font-mono">Edit</span> mode and change one line.</li>
              <li>Pay attention to shapes: print <span className="font-mono">.shape</span> whenever you are unsure about matrix dimensions.</li>
              <li>Try to connect eigenvalues/eigenvectors to where variance is largest in the dataset after PCA.</li>
              <li>Recreate these examples from scratch in a separate notebook without copy-pasting; this will solidify the concepts.</li>
            </ul>
          </div>
        </Section>
      </main>
    </div>
  );
}

export default App;
