import React, { useState, useRef } from "react";
import axios from "axios";
import Webcam from "react-webcam";

const API_URL = "http://127.0.0.1:8000/face";

const FaceUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [useWebcam, setUseWebcam] = useState<boolean>(false);

  const webcamRef = useRef<Webcam>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const captureWebcam = () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    // Convert base64 to File
    const byteString = atob(imageSrc.split(",")[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
    const blob = new Blob([ab], { type: "image/png" });
    const webcamFile = new File([blob], "webcam.png", { type: "image/png" });
    setFile(webcamFile);
  };

  const handleRegister = async () => {
    if (!file || !name) {
      alert("Please enter a name and select a file or capture from webcam.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name);

    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/register`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err: any) {
      console.error(err);
      alert(err.response?.data?.detail || "Error registering face");
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async () => {
    if (!file) {
      alert("Please select a file or capture from webcam.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/verify`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err: any) {
      console.error(err);
      alert(err.response?.data?.detail || "Error verifying face");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h1 className="text-xl font-bold mb-4">FaceKey API Test</h1>

      <input
        type="text"
        placeholder="Enter name for registration"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="mb-2 p-2 border rounded w-full"
      />

      <div className="mb-2">
        <label className="mr-2">
          <input
            type="radio"
            checked={!useWebcam}
            onChange={() => setUseWebcam(false)}
          />
          Upload Image
        </label>
        <label>
          <input
            type="radio"
            checked={useWebcam}
            onChange={() => setUseWebcam(true)}
          />
          Use Webcam
        </label>
      </div>

      {!useWebcam && (
        <input type="file" onChange={handleFileChange} accept="image/*" />
      )}
      {useWebcam && (
        <div className="mb-2">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/png"
            width={320}
            height={240}
          />
          <button
            className="ml-2 px-4 py-2 bg-yellow-500 text-white rounded"
            onClick={captureWebcam}
          >
            Capture
          </button>
        </div>
      )}

      <div className="flex space-x-2 mb-4">
        <button
          onClick={handleRegister}
          className="px-4 py-2 bg-green-500 text-white rounded"
          disabled={loading}
        >
          {loading ? "Processing..." : "Register Face"}
        </button>

        <button
          onClick={handleVerify}
          className="px-4 py-2 bg-blue-500 text-white rounded"
          disabled={loading}
        >
          {loading ? "Processing..." : "Verify Face"}
        </button>
      </div>

      {result && (
        <div className="mt-4 p-4 border rounded bg-gray-100">
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default FaceUpload;
