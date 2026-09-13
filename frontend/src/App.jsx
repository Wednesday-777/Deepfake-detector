import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Connection failed. Check that the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="console">
        <header className="console-header">
          <h1 className="brand">deepfake_detector</h1>
          <p className="tagline">
            Upload a face image. The model scans it for signs of AI generation
            and shows you exactly what it looked at.
          </p>
        </header>

        <div
          className={`dropzone ${selectedFile ? 'dropzone-filled' : ''}`}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            id="file-input"
          />
          <label htmlFor="file-input" className="dropzone-label">
            <span className="dropzone-icon">▲</span>
            <span>{selectedFile ? selectedFile.name : 'select an image to scan'}</span>
          </label>
        </div>

        <button
          onClick={handleSubmit}
          disabled={!selectedFile || loading}
          className="scan-button"
        >
          {loading ? 'scanning…' : 'run scan'}
        </button>

        {error && <div className="error-line">! {error}</div>}

        {(previewUrl || result) && (
          <div className="scan-grid">
            <div className="scan-panel">
              <span className="panel-label">source</span>
              {previewUrl && <img src={previewUrl} alt="Original upload" />}
            </div>

            <div className={`scan-panel ${loading ? 'scanning' : ''}`}>
              <span className="panel-label">attention map</span>
              {loading && (
                <div className="scan-placeholder">
                  <div className="scan-line" />
                </div>
              )}
              {result && !loading && (
                <img
                  src={`data:image/png;base64,${result.gradcam_image_base64}`}
                  alt="Grad-CAM overlay"
                />
              )}
            </div>
          </div>
        )}

        {result && (
          <div className="readout">
            <div className="readout-row">
              <span className="readout-key">verdict</span>
              <span className={`readout-value verdict-${result.label}`}>
                {result.label === 'fake' ? 'AI-GENERATED' : 'LIKELY REAL'}
              </span>
            </div>
            <div className="readout-row">
              <span className="readout-key">confidence</span>
              <div className="confidence-bar">
                <div
                  className={`confidence-fill fill-${result.label}`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
              <span className="readout-pct">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}

        {result && (
          <details className="limitations">
            <summary>model limitations</summary>
            <ul>
              <li>Trained on the 140k Real and Fake Faces dataset (StyleGAN-generated fakes) — may not generalize to diffusion-based generators like Gemini or Midjourney.</li>
              <li>Test performance: 95.14% accuracy, 0.9926 AUC. Recall on fakes (98%) is higher than on real images (93%), so the model leans toward flagging uncertain cases as fake rather than missing one.</li>
              <li>Built for cropped, front-facing face images — untested on full-body shots or non-face content.</li>
            </ul>
          </details>
        )}

        <footer className="console-footer">
          EfficientNetB0 + Grad-CAM · <a href="#">source</a>
        </footer>
      </div>
    </div>
  )
}

export default App
