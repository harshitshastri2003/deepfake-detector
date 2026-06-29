import { useState, useRef } from "react";

const API_URL = "https://harshitshastri2003-deepfake-detector.hf.space";

const DonutChart = ({ real, fake }) => {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const realOffset = circumference - (real / 100) * circumference;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
      <svg width="120" height="120" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={radius} fill="none" stroke="#1a1a2e" strokeWidth="12" />
        <circle
          cx="50" cy="50" r={radius} fill="none"
          stroke="#ef4444" strokeWidth="12"
          strokeDasharray={circumference}
          strokeDashoffset={0}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
        />
        <circle
          cx="50" cy="50" r={radius} fill="none"
          stroke="#22c55e" strokeWidth="12"
          strokeDasharray={circumference}
          strokeDashoffset={realOffset}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
        <text x="50" y="45" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
          {real.toFixed(1)}%
        </text>
        <text x="50" y="60" textAnchor="middle" fill="#aaa" fontSize="8">
          Real
        </text>
      </svg>
      <div style={{ display: "flex", gap: "16px", fontSize: "12px" }}>
        <span style={{ color: "#22c55e" }}>● Real: {real.toFixed(1)}%</span>
        <span style={{ color: "#ef4444" }}>● Fake: {fake.toFixed(1)}%</span>
      </div>
    </div>
  );
};

const LoadingAnimation = () => (
  <div style={{
    display: "flex", flexDirection: "column",
    alignItems: "center", gap: "20px", padding: "40px"
  }}>
    <div style={{ position: "relative", width: "80px", height: "80px" }}>
      <div style={{
        position: "absolute", inset: 0,
        borderRadius: "50%",
        border: "3px solid transparent",
        borderTopColor: "#8b5cf6",
        animation: "spin 1s linear infinite"
      }} />
      <div style={{
        position: "absolute", inset: "8px",
        borderRadius: "50%",
        border: "3px solid transparent",
        borderTopColor: "#3b82f6",
        animation: "spin 0.8s linear infinite reverse"
      }} />
      <div style={{
        position: "absolute", inset: "16px",
        borderRadius: "50%",
        border: "3px solid transparent",
        borderTopColor: "#06b6d4",
        animation: "spin 0.6s linear infinite"
      }} />
      <div style={{
        position: "absolute", inset: "24px",
        borderRadius: "50%",
        background: "radial-gradient(circle, #8b5cf6, #3b82f6)",
        animation: "pulse 1s ease-in-out infinite"
      }} />
    </div>
    <div style={{ textAlign: "center" }}>
      <p style={{ color: "#8b5cf6", fontWeight: "600", fontSize: "16px", margin: 0 }}>
        Analysing Image
      </p>
      <p style={{ color: "#666", fontSize: "13px", margin: "4px 0 0 0" }}>
        Running EfficientNet-B4 + ViT inference...
      </p>
    </div>
    <div style={{ display: "flex", gap: "6px" }}>
      {[0, 1, 2, 3, 4].map(i => (
        <div key={i} style={{
          width: "8px", height: "8px",
          borderRadius: "50%",
          background: "#8b5cf6",
          animation: `bounce 0.8s ease-in-out ${i * 0.15}s infinite alternate`
        }} />
      ))}
    </div>
    <style>{`
      @keyframes spin { to { transform: rotate(360deg); } }
      @keyframes pulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
      @keyframes bounce { from { transform: translateY(0); } to { transform: translateY(-10px); } }
    `}</style>
  </div>
);

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const analyse = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch(`${API_URL}/detect`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
      setHistory(prev => [{
        name: image.name,
        prediction: data.prediction,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString()
      }, ...prev].slice(0, 5));
    } catch (err) {
      setError("Error connecting to backend. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const isReal = result?.prediction === "Real";

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%)",
      color: "white",
      fontFamily: "'Segoe UI', sans-serif",
      padding: "0 20px 40px"
    }}>
      {/* Header */}
      <div style={{
        textAlign: "center", padding: "50px 20px 30px",
        borderBottom: "1px solid rgba(139,92,246,0.2)"
      }}>
        <div style={{
          display: "inline-block",
          background: "rgba(139,92,246,0.1)",
          border: "1px solid rgba(139,92,246,0.3)",
          borderRadius: "20px", padding: "6px 16px",
          fontSize: "13px", color: "#8b5cf6", marginBottom: "20px"
        }}>
          ⚡ Powered by Hybrid Deep Learning Architecture
        </div>
        <h1 style={{
          fontSize: "clamp(32px, 6vw, 60px)", fontWeight: "800",
          margin: "0 0 16px",
          background: "linear-gradient(135deg, #ffffff, #8b5cf6, #3b82f6)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
        }}>
          Harshit's Deepfake Detector
        </h1>
        <p style={{ color: "#888", fontSize: "16px", maxWidth: "500px", margin: "0 auto 30px" }}>
          Upload any image and our AI will analyze it for signs of digital manipulation
        </p>

        {/* Stats */}
        <div style={{
          display: "flex", justifyContent: "center",
          gap: "40px", flexWrap: "wrap"
        }}>
          {[
            { value: "97%", label: "Model Accuracy" },
            { value: "EfficientNet+ViT", label: "Hybrid Architecture" },
            { value: "550K+", label: "Images Trained" },
            { value: "Grad-CAM", label: "Explainable AI" }
          ].map((stat, i) => (
            <div key={i} style={{ textAlign: "center" }}>
              <div style={{
                fontSize: "20px", fontWeight: "700",
                background: "linear-gradient(135deg, #8b5cf6, #3b82f6)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
              }}>{stat.value}</div>
              <div style={{ fontSize: "12px", color: "#666" }}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: "1100px", margin: "40px auto", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>

        {/* Left — Upload */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: `2px dashed ${dragging ? "#8b5cf6" : "rgba(139,92,246,0.3)"}`,
          borderRadius: "20px", padding: "30px",
          display: "flex", flexDirection: "column", alignItems: "center", gap: "20px",
          transition: "all 0.3s",
          cursor: "pointer"
        }}
          onClick={() => fileRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        >
          <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])} />

          {preview ? (
            <img src={preview} alt="upload" style={{
              width: "100%", maxHeight: "280px",
              objectFit: "contain", borderRadius: "12px"
            }} />
          ) : (
            <div style={{ textAlign: "center", padding: "40px 0" }}>
              <div style={{ fontSize: "48px", marginBottom: "12px" }}>🖼️</div>
              <p style={{ color: "#8b5cf6", fontWeight: "600", margin: "0 0 8px" }}>
                Drop image here or click to upload
              </p>
              <p style={{ color: "#555", fontSize: "13px", margin: 0 }}>
                Supports JPG, PNG, WEBP
              </p>
            </div>
          )}

          {preview && (
            <p style={{ color: "#666", fontSize: "12px", margin: 0 }}>
              Click to change image
            </p>
          )}
        </div>

        {/* Right — Result */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(139,92,246,0.2)",
          borderRadius: "20px", padding: "30px",
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: loading || result ? "flex-start" : "center",
          gap: "20px", minHeight: "300px"
        }}>
          {!loading && !result && (
            <div style={{ textAlign: "center", color: "#555" }}>
              <div style={{ fontSize: "40px", marginBottom: "12px" }}>🔍</div>
              <p>Upload an image and click Analyse</p>
            </div>
          )}

          {loading && <LoadingAnimation />}

          {error && (
            <div style={{
              background: "rgba(239,68,68,0.1)",
              border: "1px solid rgba(239,68,68,0.3)",
              borderRadius: "12px", padding: "16px",
              color: "#ef4444", textAlign: "center"
            }}>{error}</div>
          )}

          {result && !loading && (
            <>
              {/* Result Badge */}
              <div style={{
                width: "100%",
                background: isReal
                  ? "rgba(34,197,94,0.1)"
                  : "rgba(239,68,68,0.1)",
                border: `1px solid ${isReal ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)"}`,
                borderRadius: "16px", padding: "20px",
                textAlign: "center",
                boxShadow: isReal
                  ? "0 0 30px rgba(34,197,94,0.15)"
                  : "0 0 30px rgba(239,68,68,0.15)"
              }}>
                <div style={{ fontSize: "32px", marginBottom: "8px" }}>
                  {isReal ? "✅" : "🚨"}
                </div>
                <div style={{
                  fontSize: "24px", fontWeight: "800",
                  color: isReal ? "#22c55e" : "#ef4444"
                }}>
                  IMAGE IS {result.prediction.toUpperCase()}
                </div>
                <div style={{
                  fontSize: "36px", fontWeight: "900",
                  color: isReal ? "#22c55e" : "#ef4444",
                  margin: "8px 0"
                }}>
                  {result.confidence}%
                </div>
                <div style={{ color: "#888", fontSize: "13px" }}>confidence</div>
              </div>

              {/* Confidence Bar */}
              <div style={{ width: "100%" }}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  fontSize: "12px", color: "#888", marginBottom: "6px"
                }}>
                  <span>Fake</span>
                  <span>Real</span>
                </div>
                <div style={{
                  height: "8px", background: "#1a1a2e",
                  borderRadius: "4px", overflow: "hidden"
                }}>
                  <div style={{
                    height: "100%",
                    width: `${result.probabilities.Real}%`,
                    background: "linear-gradient(90deg, #ef4444, #22c55e)",
                    borderRadius: "4px",
                    transition: "width 1s ease"
                  }} />
                </div>
              </div>

              {/* Donut Chart */}
              <DonutChart
                real={result.probabilities.Real}
                fake={result.probabilities.Fake}
              />
            </>
          )}
        </div>
      </div>

      {/* Analyse Button */}
      {preview && (
        <div style={{ textAlign: "center", margin: "0 auto 40px", maxWidth: "1100px" }}>
          <button onClick={analyse} disabled={loading} style={{
            background: loading
              ? "rgba(139,92,246,0.3)"
              : "linear-gradient(135deg, #8b5cf6, #3b82f6)",
            border: "none", borderRadius: "12px",
            padding: "16px 60px", fontSize: "16px",
            fontWeight: "700", color: "white",
            cursor: loading ? "not-allowed" : "pointer",
            transition: "all 0.3s",
            boxShadow: loading ? "none" : "0 8px 30px rgba(139,92,246,0.4)"
          }}>
            {loading ? "Analysing..." : "🔍 Analyse Image"}
          </button>
        </div>
      )}

      {/* Heatmap Section */}
      {result?.heatmap && (
        <div style={{
          maxWidth: "1100px", margin: "0 auto 40px",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(139,92,246,0.2)",
          borderRadius: "20px", padding: "30px"
        }}>
          <h3 style={{
            margin: "0 0 20px",
            background: "linear-gradient(135deg, #8b5cf6, #3b82f6)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            fontWeight: "700"
          }}>
            🔥 Grad-CAM Explainability
          </h3>
          <p style={{ color: "#666", fontSize: "13px", margin: "0 0 20px" }}>
            Highlighted regions show where the model focused to make its decision
          </p>
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "20px"
          }}>
            <div style={{ textAlign: "center" }}>
              <p style={{ color: "#888", fontSize: "13px", marginBottom: "10px" }}>Original</p>
              <img src={preview} alt="original" style={{
                width: "100%", borderRadius: "12px",
                border: "1px solid rgba(139,92,246,0.2)"
              }} />
            </div>
            <div style={{ textAlign: "center" }}>
              <p style={{ color: "#888", fontSize: "13px", marginBottom: "10px" }}>Grad-CAM Heatmap</p>
              <img
                src={`data:image/jpeg;base64,${result.heatmap}`}
                alt="heatmap" style={{
                  width: "100%", borderRadius: "12px",
                  border: "1px solid rgba(139,92,246,0.2)"
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div style={{
          maxWidth: "1100px", margin: "0 auto",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(139,92,246,0.2)",
          borderRadius: "20px", padding: "30px"
        }}>
          <h3 style={{
            margin: "0 0 20px",
            background: "linear-gradient(135deg, #8b5cf6, #3b82f6)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            fontWeight: "700"
          }}>
            🕒 Analysis History
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {history.map((item, i) => (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "center",
                background: "rgba(255,255,255,0.03)",
                border: `1px solid ${item.prediction === "Real" ? "rgba(34,197,94,0.2)" : "rgba(239,68,68,0.2)"}`,
                borderRadius: "10px", padding: "12px 16px"
              }}>
                <span style={{ color: "#888", fontSize: "13px" }}>{item.name}</span>
                <span style={{
                  color: item.prediction === "Real" ? "#22c55e" : "#ef4444",
                  fontWeight: "700", fontSize: "14px"
                }}>
                  {item.prediction} — {item.confidence}%
                </span>
                <span style={{ color: "#555", fontSize: "12px" }}>{item.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{
        textAlign: "center", marginTop: "60px",
        color: "#444", fontSize: "13px"
      }}>
        Built with EfficientNet-B4 + Vision Transformer + Grad-CAM
        <br />
        <a href="https://github.com/harshitshastri2003/deepfake-detector"
          style={{ color: "#8b5cf6", textDecoration: "none" }}
          target="_blank" rel="noreferrer">
          View on GitHub
        </a>
      </div>
    </div>
  );
}