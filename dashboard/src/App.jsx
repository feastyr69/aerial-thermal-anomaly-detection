import { useEffect, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8002'

function App() {
  const [videoFile, setVideoFile] = useState(null)
  const [sourceUrl, setSourceUrl] = useState('')
  const [resultUrl, setResultUrl] = useState('')
  const [confidence, setConfidence] = useState(0.25)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState('')
  const [apiStatus, setApiStatus] = useState('checking')
  const [apiMessage, setApiMessage] = useState('Checking inference API…')

  useEffect(() => {
    let active = true

    async function checkApi() {
      try {
        const response = await fetch(`${API_URL}/api/health`)
        const health = await response.json()
        if (!response.ok || health.status !== 'ready') {
          throw new Error(health.detail || 'The model did not load. Check the API terminal and weights file.')
        }
        if (active) {
          setApiStatus('ready')
          setApiMessage('Model loaded and ready')
        }
      } catch (healthError) {
        if (active) {
          setApiStatus('offline')
          setApiMessage(healthError.message === 'Failed to fetch'
            ? 'Inference API unavailable. Activate .venv and run: python thermal-streaming/inference_api.py'
            : healthError.message)
        }
      }
    }

    checkApi()
    const timer = window.setInterval(checkApi, 5000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (!videoFile) {
      setSourceUrl('')
      return undefined
    }

    const url = URL.createObjectURL(videoFile)
    setSourceUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [videoFile])

  useEffect(() => () => {
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  }, [resultUrl])

  function selectVideo(event) {
    setVideoFile(event.target.files?.[0] ?? null)
    setResultUrl('')
    setError('')
  }

  async function runInference(event) {
    event.preventDefault()
    if (!videoFile || processing || apiStatus !== 'ready') return

    setProcessing(true)
    setError('')
    setResultUrl('')

    const form = new FormData()
    form.append('video', videoFile)
    form.append('confidence', String(confidence))

    try {
      const response = await fetch(`${API_URL}/api/infer`, { method: 'POST', body: form })
      if (!response.ok) {
        let message = `Inference failed (${response.status})`
        try {
          const payload = await response.json()
          message = payload.detail || message
        } catch {
          // Keep the status message if the server did not return JSON.
        }
        throw new Error(message)
      }

      const resultBlob = await response.blob()
      setResultUrl(URL.createObjectURL(resultBlob))
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the inference API.')
    } finally {
      setProcessing(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Thermal Vision home">
          <span className="brand-mark" aria-hidden="true">T</span>
          <span>THERMAL<span className="brand-muted">VISION</span></span>
        </a>
        <div className={`system-state state-${apiStatus}`}><span className="state-dot" /> {apiStatus === 'ready' ? 'MODEL READY' : apiStatus === 'checking' ? 'CONNECTING' : 'API OFFLINE'}</div>
      </header>

      <section className="intro" id="top">
        <p className="eyebrow">MODEL TEST WORKSPACE <span> / </span> HIT-UAV</p>
        <h1>Thermal video<br /><em>inference</em></h1>
        <p className="intro-copy">Upload a clip to inspect YOLO detections frame by frame. The original and annotated videos stay side by side for direct comparison.</p>
      </section>

      {apiStatus !== 'ready' && <div className={`api-banner api-${apiStatus}`} role="status">{apiMessage}</div>}

      <form className="control-panel" onSubmit={runInference}>
        <label className="file-picker">
          <input type="file" accept="video/*,.mkv,.avi,.mov,.mp4,.webm" onChange={selectVideo} disabled={processing} />
          <span className="upload-icon" aria-hidden="true">↑</span>
          <span className="file-copy">
            <strong>{videoFile ? videoFile.name : 'Choose a video file'}</strong>
            <small>{videoFile ? `${(videoFile.size / (1024 * 1024)).toFixed(1)} MB · ready to process` : 'MP4, MOV, AVI, MKV, or WebM'}</small>
          </span>
          <span className="browse-button">Browse</span>
        </label>

        <div className="run-controls">
          <label className="confidence-control" htmlFor="confidence">
            <span>CONFIDENCE <strong>{confidence.toFixed(2)}</strong></span>
            <input
              id="confidence"
              type="range"
              min="0.05"
              max="0.85"
              step="0.05"
              value={confidence}
              onChange={(event) => setConfidence(Number(event.target.value))}
              disabled={processing}
            />
          </label>
          <button className="run-button" type="submit" disabled={!videoFile || processing || apiStatus !== 'ready'}>
            {processing ? <><span className="spinner" /> Processing video</> : apiStatus !== 'ready' ? <>API unavailable</> : <>Run inference <span aria-hidden="true">→</span></>}
          </button>
        </div>
      </form>

      {error && <div className="error-banner" role="alert"><span>!</span>{error}</div>}

      <section className="comparison" aria-label="Video comparison">
        <VideoPanel title="SOURCE VIDEO" status={videoFile ? 'INPUT' : 'AWAITING VIDEO'} src={sourceUrl} />
        <VideoPanel title="YOLO INFERENCE" status={resultUrl ? 'ANNOTATED OUTPUT' : processing ? 'PROCESSING' : 'OUTPUT'} src={resultUrl} processing={processing} />
      </section>

      <footer className="footer-note">
        <span>YOLOv8 · UAVBEST.PT</span>
        <span>Predictions reflect model output; confidence is adjustable.</span>
      </footer>
    </main>
  )
}

function VideoPanel({ title, status, src, processing = false }) {
  return (
    <article className="video-panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <span className={`panel-status ${src ? 'is-ready' : ''}`}><i />{status}</span>
      </div>
      <div className="video-stage">
        {src ? (
          <video src={src} controls playsInline preload="metadata" />
        ) : (
          <div className="empty-stage">
            {processing ? <span className="large-spinner" /> : <span className="play-symbol" aria-hidden="true">▶</span>}
            <p>{processing ? 'Running inference on each frame…' : title === 'SOURCE VIDEO' ? 'Select a video to preview it here' : 'Annotated video will appear here'}</p>
          </div>
        )}
      </div>
    </article>
  )
}

export default App
