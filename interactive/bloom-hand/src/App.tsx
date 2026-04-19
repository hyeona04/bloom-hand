import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Flower2, Loader2, Sparkles } from "lucide-react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

type Bloom = {
  id: number;
  x: number;
  y: number;
  size: number;
  rotation: number;
  createdAt: number;
  petals: number;
  isClosing: boolean;
};

type Landmark = {
  x: number;
  y: number;
  z: number;
};

const BLOOM_LIFETIME = 2600;
const HAND_SCORE_THRESHOLD = 0.45;
const BACKGROUND_IMAGE = "/flower-bg.jpg";

function distance(a: Landmark, b: Landmark) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function estimatePalmOpen(hand: Landmark[]) {
  if (!hand || hand.length < 21) return false;

  const wrist = hand[0];
  const thumbTip = hand[4];
  const indexTip = hand[8];
  const middleTip = hand[12];
  const ringTip = hand[16];
  const pinkyTip = hand[20];

  const palmCenter = {
    x: (wrist.x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5,
    y: (wrist.y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5,
    z: (wrist.z + hand[5].z + hand[9].z + hand[13].z + hand[17].z) / 5,
  };

  const openness =
    [thumbTip, indexTip, middleTip, ringTip, pinkyTip]
      .map((tip) => distance(tip, palmCenter))
      .reduce((a, b) => a + b, 0) / 5;

  const palmWidth = distance(hand[5], hand[17]);
  return openness > palmWidth * 0.95;
}

function drawFlower(ctx: CanvasRenderingContext2D, bloom: Bloom, age: number) {
  const rawProgress = Math.min(age / 700, 1);
  const openProgress = bloom.isClosing ? 1 - rawProgress : rawProgress;
  const fade = Math.max(1 - age / BLOOM_LIFETIME, 0);
  const size = bloom.size * (0.2 + openProgress * 0.8);

  ctx.save();
  ctx.translate(bloom.x, bloom.y);
  ctx.rotate((bloom.rotation * Math.PI) / 180);
  ctx.globalAlpha = fade;

  for (let i = 0; i < bloom.petals; i++) {
    const angle = (Math.PI * 2 * i) / bloom.petals;
    ctx.save();
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, -size * 0.46, size * 0.2, size * 0.44, 0, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${48 + i * 2}, 78%, ${84 - i * 1.3}%)`;
    ctx.fill();
    ctx.restore();
  }

  ctx.beginPath();
  ctx.arc(0, 0, size * 0.12, 0, Math.PI * 2);
  ctx.fillStyle = "#e2c400";
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(0, size * 0.08);
  ctx.quadraticCurveTo(size * 0.08, size * 0.45, 0, size * 1.1);
  ctx.strokeStyle = "rgba(92, 132, 34, 0.85)";
  ctx.lineWidth = Math.max(size * 0.04, 1.8);
  ctx.stroke();

  ctx.restore();
}

function drawSparkles(ctx: CanvasRenderingContext2D, x: number, y: number, time: number) {
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI * 2 * i) / 6 + time / 700;
    const r = 18 + (i % 2) * 8;
    const sx = x + Math.cos(angle) * r;
    const sy = y + Math.sin(angle) * r;

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(angle);
    ctx.globalAlpha = 0.4;
    ctx.strokeStyle = "rgba(255,255,210,0.9)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(-3, 0);
    ctx.lineTo(3, 0);
    ctx.moveTo(0, -3);
    ctx.lineTo(0, 3);
    ctx.stroke();
    ctx.restore();
  }
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const backgroundRef = useRef<HTMLImageElement | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const bloomIdRef = useRef<number>(0);
  const lastBloomTimeRef = useRef<number>(0);

  const [loading, setLoading] = useState(true);
  const [tracking, setTracking] = useState(false);
  const [status, setStatus] = useState("사진 위에 피어나는 꽃 인터랙션을 준비 중입니다.");
  const [permissionError, setPermissionError] = useState<string | null>(null);

  const bloomsRef = useRef<Bloom[]>([]);
  const handCenterRef = useRef<{ x: number; y: number; visible: boolean }>({
    x: 0,
    y: 0,
    visible: false,
  });

  useEffect(() => {
    let stream: MediaStream | null = null;

    const setup = async () => {
      try {
        setLoading(true);
        setStatus("배경 이미지와 손 인식 모델을 준비하는 중입니다.");

        const img = new Image();
        img.src = BACKGROUND_IMAGE;
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("배경 이미지를 불러오지 못했습니다."));
        });
        backgroundRef.current = img;

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          numHands: 1,
          runningMode: "VIDEO",
          minHandDetectionConfidence: HAND_SCORE_THRESHOLD,
          minHandPresenceConfidence: HAND_SCORE_THRESHOLD,
          minTrackingConfidence: HAND_SCORE_THRESHOLD,
        });

        setTracking(true);
        setStatus("카메라는 숨겨지고, 손동작만 인식합니다. 손바닥을 펴면 꽃이 피어요.");
        setLoading(false);
      } catch (error) {
        console.error(error);
        setPermissionError("카메라 권한 또는 배경 이미지 로딩에 실패했습니다.");
        setStatus("실행할 수 없습니다.");
        setLoading(false);
      }
    };

    setup();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      handLandmarkerRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!tracking) return;

    const render = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const handLandmarker = handLandmarkerRef.current;
      const bg = backgroundRef.current;

      if (!video || !canvas || !handLandmarker || !bg) {
        animationRef.current = requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        animationRef.current = requestAnimationFrame(render);
        return;
      }

      if (canvas.width !== bg.width || canvas.height !== bg.height) {
        canvas.width = bg.width;
        canvas.height = bg.height;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);

      const now = performance.now();

      if (video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const result = handLandmarker.detectForVideo(video, now);
        const hand = result.landmarks?.[0];

        if (hand && hand.length >= 21) {
          const palmOpen = estimatePalmOpen(hand as Landmark[]);

          const palmX = (1-(hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5) * canvas.width;
          const palmY = ((hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5) * canvas.height;

          handCenterRef.current = { x: palmX, y: palmY, visible: true };

          if (palmOpen && now - lastBloomTimeRef.current > 240) {
            lastBloomTimeRef.current = now;
            bloomsRef.current.push({
              id: bloomIdRef.current++,
              x: palmX + (Math.random() * 30 - 15),
              y: palmY + (Math.random() * 30 - 15),
              size: 30 + Math.random() * 20,
              rotation: Math.random() * 360,
              createdAt: now,
              petals: 6 + Math.floor(Math.random() * 3),
              isClosing: Math.random() > 0.5,
            });
          }
        } else {
          handCenterRef.current.visible = false;
        }
      }

      const aliveBlooms: Bloom[] = [];
      for (const bloom of bloomsRef.current) {
        const age = now - bloom.createdAt;
        if (age < BLOOM_LIFETIME) {
          drawFlower(ctx, bloom, age);
          aliveBlooms.push(bloom);
        }
      }
      bloomsRef.current = aliveBlooms;

      if (handCenterRef.current.visible) {
        drawSparkles(ctx, handCenterRef.current.x, handCenterRef.current.y, now);
      }

      ctx.save();
      ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
      ctx.fillRect(0, canvas.height - 70, canvas.width, 70);
      ctx.fillStyle = "#fffde8";
      ctx.font = "600 24px sans-serif";
      ctx.fillText("Show your palm, the flowers breathe.", 24, canvas.height - 28);
      ctx.restore();

      animationRef.current = requestAnimationFrame(render);
    };

    animationRef.current = requestAnimationFrame(render);

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [tracking]);

  return (
    <div className="min-h-screen bg-[#1e2d09] text-white">
      <div className="mx-auto grid min-h-screen max-w-7xl gap-6 p-6 lg:grid-cols-[360px_1fr]">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col gap-6"
        >
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="rounded-2xl bg-yellow-300/10 p-3 text-yellow-100">
                <Flower2 className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Quiet Bloom</h1>
                <p className="text-sm text-white/70">카메라는 숨기고, 사진 위에서 꽃만 피고 지는 인터랙션</p>
              </div>
            </div>

            <div className="space-y-3 text-sm text-white/80">
              <p>이 버전은 웹캠 화면을 직접 보여주지 않습니다.</p>
              <p>사용자의 손동작은 백그라운드에서만 인식하고, 화면에는 업로드한 꽃 사진과 애니메이션만 보입니다.</p>
              <p>Figma는 이 화면의 레이아웃과 분위기 시안을 만드는 데 쓰고, 실제 인터랙션은 React에서 구현합니다.</p>
            </div>

            <div className="mt-5 grid gap-3">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                  <Sparkles className="h-4 w-4" /> 상태
                </div>
                <p className="text-sm text-white/75">{status}</p>
              </div>
            </div>

            {permissionError && (
              <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                {permissionError}
              </div>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
          className="relative overflow-hidden rounded-[32px] border border-white/10 bg-black/30 shadow-2xl"
        >
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={canvasRef} className="h-full min-h-[640px] w-full object-cover" />

          {loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-950/80 backdrop-blur-sm">
              <Loader2 className="h-10 w-10 animate-spin" />
              <p className="text-sm text-white/80">꽃 장면을 준비하고 있습니다.</p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}


/*
중요
1. public 폴더를 만들고 flower-bg.jpg 이름으로 꽃 사진을 넣기
2. 현재 사용자가 올린 사진을 flower-bg.jpg로 저장해서 public 폴더에 넣기
3. 카메라는 hidden 상태이고, 손 인식만 백그라운드에서 수행됨
4. 따라서 화면에는 카메라 영상 대신 꽃 사진만 보임

Figma 사용 방식
- Figma에서 먼저 메인 화면 시안 제작
- 예: 전체 배경, 문구 위치, 버튼 위치, 감성적인 타이포 스타일
- 그 다음 React 코드에서 실제 손 인식 및 꽃 애니메이션 연결

즉
Figma = 디자인
React + MediaPipe = 인터랙션 구현
*/
