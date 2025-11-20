import React, { useState, useRef, useCallback } from "react";
import axios from "axios";
import Webcam from "react-webcam";
import illustrationImage from "../assets/inter.png";

const CameraIcon = () => <span className="btn-secondary-icon">📸</span>;
const QuestionIcon = () => <span className="btn-secondary-icon">❓</span>;

const API_URL = import.meta.env.VITE_API_URL;
console.log("API URL:", API_URL);

const FaceUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showHowToUse, setShowHowToUse] = useState<boolean>(false);

  const webcamRef = useRef<Webcam>(null);

  /** Capture webcam → convert to File */
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
  }, []);

  /** REGISTER */
  const handleRegister = async () => {
    if (!file || !name) {
      alert("Veuillez capturer une image et entrer un nom.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name);

    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/register`, formData);
      setResult(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Erreur lors de l'enregistrement");
    } finally {
      setLoading(false);
    }
  };

  /** VERIFY */
  const handleVerify = async () => {
    if (!file) {
      alert("Veuillez capturer une image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/verify`, formData);
      setResult(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Erreur lors de la vérification");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-layout-container">
      {/* LEFT PANEL */}
      <div className="form-section">
        <div className="face-upload-card">

          <h1 className="header-title">
            <span className="header-title-highlight">Face Recognition System</span><br />
            <span className="header-title-highlight">Register & Verify Identities</span>
          </h1>

          <p className="header-subtitle">FAST • RELIABLE • SECURE FACE MATCHING.</p>

          {/* --- CAMERA ONLY --- */}
          <div className="webcam-container">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/png"
              width={320}
              height={240}
              videoConstraints={{ facingMode: "user" }}
              style={{ borderRadius: "8px", maxWidth: "100%", height: "auto" }}
            />

            <button
              onClick={captureWebcam}
              className="webcam-capture-button"
            >
              Capture Photo
            </button>

            {file && <p className="file-selected-text">Image capturée ✔</p>}
          </div>

          <div className="action-buttons-group">
            <button
              onClick={() => setShowHowToUse(!showHowToUse)}
              className="btn-secondary"
            >
              <QuestionIcon /> How to use it?
            </button>
          </div>

          {showHowToUse && (
            <div className="how-to-use-text">
              <p><strong>Comment utiliser :</strong></p>
              <ol>
                <li>Activez la caméra (déjà affichée).</li>
                <li>Placez votre visage bien éclairé et centré.</li>
                <li>Cliquez sur <b>Capture Photo</b>.</li>
                <li>Pour <b>Register</b>, saisissez un nom.</li>
                <li>Pour <b>Verify</b>, laissez le nom vide.</li>
              </ol>
            </div>
          )}

          <input
            type="text"
            placeholder="Enter name for registration (optional for verify)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="input-name"
          />

          <div className="btn-group-actions">
            <button onClick={handleRegister} disabled={loading || !file} className="btn-register">
              {loading ? "Processing..." : "Register Face"}
            </button>

            <button onClick={handleVerify} disabled={loading || !file} className="btn-verify">
              {loading ? "Processing..." : "Verify Face"}
            </button>
          </div>

          {result && (
            <div className="api-result-box">
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div className="illustration-section">
        <img src={illustrationImage} alt="Face Illustration" className="illustration-img" />
      </div>
    </div>
  );
};

export default FaceUpload;
