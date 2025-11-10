import React, { useState, useRef } from "react";
import axios from "axios";
import Webcam from "react-webcam";

const API_URL = "http://127.0.0.1:8000/face";

const FaceWebcam: React.FC = () => {
  const [name, setName] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const webcamRef = useRef<Webcam>(null);

  const captureWebcam = (): File | null => {
    if (!webcamRef.current) return null;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return null;

    // Convert base64 to File
    const byteString = atob(imageSrc.split(",")[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
    const blob = new Blob([ab], { type: "image/png" });
    return new File([blob], "webcam.png", { type: "image/png" });
  };

  const handleRegister = async () => {
    const file = captureWebcam();
    if (!file || !name) {
      alert("Please enter a name and capture your face.");
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
    const file = captureWebcam();
    if (!file) {
      alert("Please capture your face.");
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
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 to-black text-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6 text-cyan-400">FaceKey Authetification </h1>

      <input
        type="text"
        placeholder="Enter name for registration"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="mb-4 p-3 rounded bg-gray-800 border border-gray-700 w-full text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
      />

      <div className="mb-4 flex flex-col items-center">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/png"
          width={320}
          height={240}
          className="rounded-lg border-2 border-cyan-400 shadow-lg"
        />
        <button
          onClick={captureWebcam}
          className="mt-3 px-6 py-2 bg-cyan-500 hover:bg-cyan-600 rounded text-black font-semibold transition"
        >
          Capture
        </button>
      </div>

      <div className="flex space-x-4 mb-6">
        <button
          onClick={handleRegister}
          className={`px-6 py-2 rounded font-semibold transition ${
            loading ? "bg-gray-700 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
          }`}
          disabled={loading}
        >
          {loading ? "Processing..." : "Register Face"}
        </button>

        <button
          onClick={handleVerify}
          className={`px-6 py-2 rounded font-semibold transition ${
            loading ? "bg-gray-700 cursor-not-allowed" : "bg-blue-500 hover:bg-blue-600"
          }`}
          disabled={loading}
        >
          {loading ? "Processing..." : "Verify Face"}
        </button>
      </div>

      {result && (
        <div className="w-full max-w-md p-4 bg-gray-800 rounded shadow-inner overflow-auto">
          <h2 className="text-lg font-semibold text-cyan-400 mb-2">Result:</h2>
          <pre className="text-sm text-gray-300">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default FaceWebcam;
