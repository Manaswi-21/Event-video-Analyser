import { useEffect, useState } from "react";
import { ScanLine } from "lucide-react";
import api from "../services/api";

export default function DetectionViewer({ imageUrl, detections = [] }) {
  const [protectedImageUrl, setProtectedImageUrl] = useState("");
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl = "";

    if (!imageUrl?.startsWith("/api/media/")) {
      setProtectedImageUrl(imageUrl || "");
      setImageError(false);
      return undefined;
    }

    setProtectedImageUrl("");
    setImageError(false);
    const apiPath = imageUrl.replace(/^\/api(?=\/)/, "");
    api.get(apiPath, { responseType: "blob" })
      .then(({ data }) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(data);
        setProtectedImageUrl(objectUrl);
      })
      .catch(() => {
        if (active) setImageError(true);
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [imageUrl]);

  const displayImageUrl = imageUrl?.startsWith("/api/media/")
    ? protectedImageUrl
    : imageUrl;

  return <div className="detection-panel">
    {displayImageUrl ? <div className="detection-image-wrap"><img src={displayImageUrl} alt="Analysed frame"/>{detections.map((d, i) => {
      const [x1,y1,x2,y2] = d.bbox || d.bbox_xyxy || [];
      if ([x1,y1,x2,y2].some(v => typeof v !== "number")) return null;
      return <div key={i} className="bbox" style={{left:`${x1}%`,top:`${y1}%`,width:`${x2-x1}%`,height:`${y2-y1}%`}}><span>{d.class_name || d.label || "Object"} · {Math.round((d.confidence || 0)*100)}%</span></div>;
    })}</div> : <div className="empty-preview"><ScanLine size={28}/><span>{imageError ? "The annotated frame could not be loaded." : imageUrl ? "Loading annotated frame…" : "Annotated frames will appear here after backend analysis is connected."}</span></div>}
  </div>;
}
