#!/usr/bin/env bash
# تشغيل عقدة بذرة Living Mesh على Termux (أندرويد) + تعريضها للإنترنت مجاناً
# عبر Cloudflare Tunnel (بدون بطاقة ائتمان، بدون أي منصة استضافة سحابية).
#
# الاستخدام:
#   pkg install cloudflared        # مرة واحدة فقط، إذا لم يكن مثبتاً
#   ./scripts/run_mesh_seed_termux.sh
#
# ملاحظات مهمة:
# - هذا يستخدم "Quick Tunnel" المجاني بدون حساب Cloudflare — الرابط عشوائي
#   (*.trycloudflare.com) ويتغيّر في كل مرة تشغّل فيها السكربت من جديد.
#   لو احتجت رابطاً ثابتاً دائماً، لازم حساب Cloudflare مجاني + دومين مربوط
#   به (راجعني لو احتجت هذا لاحقاً).
# - لازم الجهاز يبقى شغّال ومتصل بالإنترنت وTermux بالخلفية طول فترة تشغيل
#   العقدة. لتجنّب إغلاق أندرويد لـTermux بالخلفية: فعّل "batch optimization
#   off" لتطبيق Termux من إعدادات النظام، واستخدم `termux-wake-lock`.
# - أوقف كل شيء بـ Ctrl+C — السكربت يقفل العملية والنفق معاً بأمان.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-7860}"
NODE_ID="${NODE_ID:-mesh_seed_termux}"
LOG_DIR="$ROOT/artifacts/living_mesh/logs"
mkdir -p "$LOG_DIR"
NODE_LOG="$LOG_DIR/node_${NODE_ID}.log"
TUNNEL_LOG="$LOG_DIR/cloudflared_${NODE_ID}.log"
URL_FILE="$ROOT/mesh_seed_tunnel_url.txt"

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "❌ cloudflared غير مثبّت. نفّذ أولاً: pkg install cloudflared" >&2
    exit 1
fi

if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock
    echo "🔒 termux-wake-lock مفعّل (يمنع أندرويد من إيقاف Termux بالخلفية)."
fi

cleanup() {
    echo ""
    echo "🛑 إيقاف العقدة والنفق..."
    [ -n "${NODE_PID:-}" ] && kill "$NODE_PID" 2>/dev/null
    [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null
    command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock
    rm -f "$URL_FILE"
}
trap cleanup EXIT INT TERM

echo "🚀 تشغيل عقدة البذرة (NODE_ID=$NODE_ID، PORT=$PORT)..."
NODE_ID="$NODE_ID" PORT="$PORT" nohup bash "$ROOT/scripts/run_mesh_seed.sh" > "$NODE_LOG" 2>&1 &
NODE_PID=$!

echo "⏳ انتظار جاهزية المنفذ المحلي..."
for i in $(seq 1 30); do
    if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "🌐 تشغيل Cloudflare Tunnel..."
nohup cloudflared tunnel --url "http://localhost:$PORT" > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

echo "⏳ انتظار رابط النفق العام..."
TUNNEL_URL=""
for i in $(seq 1 30); do
    TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -n1 || true)
    [ -n "$TUNNEL_URL" ] && break
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ لم يظهر رابط النفق بعد 30 ثانية. راجع السجل: $TUNNEL_LOG" >&2
else
    HOST_ONLY="${TUNNEL_URL#https://}"
    echo ""
    echo "✅ العقدة شغّالة وظاهرة للإنترنت على:"
    echo "   $TUNNEL_URL/status"
    echo ""
    echo "📋 بإعدادات Streamlit Cloud (Secrets) أضف:"
    echo "   NSM_ENABLE_NODE = true"
    echo "   SEED_NODE_URL = ${HOST_ONLY}:443"
    echo ""
    echo "$TUNNEL_URL" > "$URL_FILE"
fi

echo "📄 سجل العقدة: $NODE_LOG"
echo "📄 سجل النفق:  $TUNNEL_LOG"
echo "اترك هذه النافذة مفتوحة (أو استخدم tmux/screen) — اضغط Ctrl+C لإيقاف كل شيء."

wait "$NODE_PID" "$TUNNEL_PID"
