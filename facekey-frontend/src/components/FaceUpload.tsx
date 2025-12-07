import React, { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import Webcam from "react-webcam";
import "../index.css";

const API_URL = import.meta.env.VITE_API_URL;

// Friendly messages for detected emotions
const emotionMessages: Record<string, string> = {
  happy: "You're shining like a supernova 😄🌟!",
  sad: "Sending virtual hugs... 🤗💙",
  angry: "Woah! Someone needs a cookie 🍪😡😂",
  surprised: "Surprise level: OVER 9000 😱⚡",
  neutral: "Calm and steady like a wise monk 🧘‍♂️✨",
  fear: "Don't worry, you're safe with me 👻❤️",
  disgust: "Yikes! Something smells weird 🤢🤣",
};

const emotionEmojis: Record<string, string[]> = {
  happy: ["😄", "🌟", "✨", "🎉", "💫", "🎊", "🥳", "😊", "💛", "🌈"],
  sad: ["😢", "💙", "🤗", "💔", "😔", "🌧️", "😿", "💧", "🥺", "😞"],
  angry: ["😡", "🍪", "💢", "😤", "🔥", "😠", "💥", "⚡", "👿", "🌋"],
  surprised: ["😱", "⚡", "🤯", "💥", "✨", "😲", "🎆", "💫", "🌟", "🔮"],
  neutral: ["😐", "🧘‍♂️", "✨", "☮️", "🕊️", "🌿", "😶", "🤐", "💭", "🌙"],
  fear: ["😧", "👻", "❤️", "🛡️", "🌙", "😨", "😰", "🙀", "💀", "🌑"],
  disgust: ["🤢", "😖", "🤮", "😷", "🙊", "🤧", "😵", "🤒", "💚", "🦠"],
};

const FaceUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);
  const [showEmotionAnimation, setShowEmotionAnimation] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<string>("");

  const webcamRef = useRef<Webcam>(null);

  // Play sound effect based on emotion
  const playEmotionSound = (emotion: string) => {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      const emotionSounds: Record<string, { freq: number; duration: number }[]> = {
        happy: [
          { freq: 523.25, duration: 0.15 },
          { freq: 659.25, duration: 0.15 },
          { freq: 783.99, duration: 0.25 }
        ],
        sad: [
          { freq: 329.63, duration: 0.25 },
          { freq: 293.66, duration: 0.35 }
        ],
        angry: [
          { freq: 110, duration: 0.2 },
          { freq: 98, duration: 0.25 }
        ],
        surprised: [
          { freq: 880, duration: 0.12 },
          { freq: 1046.5, duration: 0.18 }
        ],
        neutral: [
          { freq: 440, duration: 0.25 }
        ],
        fear: [
          { freq: 415.3, duration: 0.12 },
          { freq: 369.99, duration: 0.18 }
        ],
        disgust: [
          { freq: 196, duration: 0.25 }
        ]
      };

      const sounds = emotionSounds[emotion.toLowerCase()] || emotionSounds.neutral;
      let currentTime = audioContext.currentTime;

      sounds.forEach((sound) => {
        oscillator.frequency.setValueAtTime(sound.freq, currentTime);
        gainNode.gain.setValueAtTime(0.3, currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, currentTime + sound.duration);
        currentTime += sound.duration;
      });

      oscillator.start(audioContext.currentTime);
      oscillator.stop(currentTime);
    } catch (error) {
      console.log("Audio context not available");
    }
  };

  // Trigger emotion animation
  useEffect(() => {
    if (result?.emotion) {
      setCurrentEmotion(result.emotion.emotion.toLowerCase());
      setShowEmotionAnimation(true);
      playEmotionSound(result.emotion.emotion);

      // Hide animation after 10 seconds
      const timer = setTimeout(() => {
        setShowEmotionAnimation(false);
      }, 10000);

      return () => clearTimeout(timer);
    }
  }, [result]);

  // Capture face from webcam
  const captureWebcam = useCallback(() => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    const byteString = atob(imageSrc.split(",")[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);

    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    const blob = new Blob([ab], { type: "image/png" });
    const webcamFile = new File([blob], "webcam.png", { type: "image/png" });

    setFile(webcamFile);
    setCameraEnabled(false);
  }, []);

  // Register a face
  const handleRegister = async () => {
    if (!file || !name) return alert("Capture a photo and enter a name!");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name);

    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/register`, formData);
      setResult(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Register Error");
    } finally {
      setLoading(false);
    }
  };

  // Verify a face
  const handleVerify = async () => {
    if (!file) return alert("Capture a photo first!");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/verify`, formData);
      setResult(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Verify Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="facekey-container">
      {/* Full Page Emoji Animation */}
      {showEmotionAnimation && currentEmotion && (
        <div className="emoji-explosion">
          {Array.from({ length: 50 }).map((_, i) => {
            const emojis = emotionEmojis[currentEmotion] || emotionEmojis.neutral;
            const randomEmoji = emojis[Math.floor(Math.random() * emojis.length)];
            return (
              <div
                key={i}
                className="floating-emoji"
                style={{
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`,
                  animationDuration: `${4 + Math.random() * 4}s`,
                  fontSize: `${2 + Math.random() * 3}rem`,
                }}
              >
                {randomEmoji}
              </div>
            );
          })}
        </div>
      )}

      {/* Confetti Effect */}
      {showEmotionAnimation && currentEmotion === "happy" && (
        <div className="confetti-container">
          {Array.from({ length: 100 }).map((_, i) => (
            <div
              key={i}
              className="confetti"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                backgroundColor: ['#ff0', '#f0f', '#0ff', '#f00', '#0f0', '#00f'][Math.floor(Math.random() * 6)],
              }}
            />
          ))}
        </div>
      )}

      <div className="facekey-card">
        {/* Header */}
        <div className="facekey-header">
          <h1 className="facekey-title">
            Face<span className="facekey-title-accent">Key</span>
          </h1>
          <p className="facekey-subtitle">FAST • RELIABLE • SECURE</p>
        </div>

        {/* Camera Section */}
        <div className="camera-section">
          {!cameraEnabled ? (
            <div className="camera-placeholder">
              <div className="camera-icon">
                <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <button className="open-camera-btn" onClick={() => setCameraEnabled(true)}>
                Open Camera
              </button>
              {file && (
                <div className="photo-captured">
                  <span className="check-icon">✓</span> Photo captured successfully
                </div>
              )}
            </div>
          ) : (
            <div className="webcam-active">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/png"
                className="webcam-video"
                videoConstraints={{ facingMode: "user" }}
              />
              <div className="webcam-controls">
                <button className="capture-btn" onClick={captureWebcam}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <circle cx="12" cy="12" r="10" />
                  </svg>
                  Capture
                </button>
                <button className="close-btn" onClick={() => setCameraEnabled(false)}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Close
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Instructions Toggle */}
        <button
          className="instructions-toggle"
          onClick={() => setShowInstructions(!showInstructions)}
        >
          {showInstructions ? "Hide" : "Show"} Instructions
        </button>

        {/* Instructions */}
        {showInstructions && (
          <div className="instructions-box">
            <h3>How to use:</h3>
            <ol>
              <li>Click "Open Camera" to activate your webcam</li>
              <li>Position your face centered and well-lit</li>
              <li>Click "Capture" to take a photo</li>
              <li>For <strong>Register</strong>: Enter your name below</li>
              <li>For <strong>Verify</strong>: Leave name empty</li>
            </ol>
          </div>
        )}

        {/* Name Input */}
        <input
          type="text"
          placeholder="Enter name (for registration only)"
          className="name-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        {/* Action Buttons */}
        <div className="action-buttons">
          <button
            className="register-btn"
            onClick={handleRegister}
            disabled={!file || loading}
          >
            {loading ? "Processing..." : "Register"}
          </button>

          <button
            className="verify-btn"
            onClick={handleVerify}
            disabled={!file || loading}
          >
            {loading ? "Processing..." : "Verify"}
          </button>
        </div>

        {/* Result Box */}
        {result && (
          <div className="result-box">
            {result.message ? (
              <>
                <h3 className="result-title">✅ Registration Result</h3>
                <p className="result-message">{result.message}</p>
              </>
            ) : result.verified !== undefined ? (
              <>
                <h3 className="result-title">🔍 Verification Result</h3>
                <div className="result-details">
                  <p>
                    <strong>Verified:</strong>{" "}
                    {result.verified ? (
                      <span className="verified-yes">✓ Yes</span>
                    ) : (
                      <span className="verified-no">✗ No</span>
                    )}
                  </p>
                  <p><strong>User:</strong> {result.user || "-"}</p>
                  <p><strong>Similarity:</strong> {result.similarity ?? "-"}</p>
                </div>

                {result.emotion && (
                  <div className="emotion-box">
                    <h3 className="emotion-title">🎭 Emotion Detected</h3>
                    <div className="emotion-emoji-main">
                      {result.emotion.emotion.toLowerCase() === "happy" && "😄"}
                      {result.emotion.emotion.toLowerCase() === "sad" && "😢"}
                      {result.emotion.emotion.toLowerCase() === "angry" && "😡"}
                      {result.emotion.emotion.toLowerCase() === "surprised" && "😱"}
                      {result.emotion.emotion.toLowerCase() === "neutral" && "😐"}
                      {result.emotion.emotion.toLowerCase() === "fear" && "😧"}
                      {result.emotion.emotion.toLowerCase() === "disgust" && "🤢"}
                    </div>
                    <p className="emotion-name">
                      <strong>{result.emotion.emotion}</strong>
                    </p>
                    <div className="confidence-bar-container">
                      <div className="confidence-bar-bg">
                        <div 
                          className="confidence-bar-fill"
                          style={{ width: `${result.emotion.confidence * 100}%` }}
                        />
                      </div>
                      <p className="emotion-confidence">
                        Confidence: {(result.emotion.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    <p className="emotion-message">
                      {emotionMessages[result.emotion.emotion.toLowerCase()]}
                    </p>
                  </div>
                )}
              </>
            ) : (
              <p>Unexpected response from API</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default FaceUpload;