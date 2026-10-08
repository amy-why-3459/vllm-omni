import asyncio
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('video_demo', str(ROOT.parent / 'realtime_duplex_demo.py'))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
original_config = demo.create_duplex_session_config
window = dict(sliding_window_mode='basic', basic_window_high_tokens=8000, basic_window_low_tokens=6000)
def config_with_window(**kwargs):
    config = original_config(**kwargs, extra_body=window)
    payload = config.to_session_payload(model='/data/why/MiniCPM-o-4_5')
    payload.pop('ref_audio', None)
    (ROOT / 'session_config.json').write_text(json.dumps(payload, indent=2))
    return config
demo.create_duplex_session_config = config_with_window
args = demo.parse_args()
result = asyncio.run(demo.run_demo(args))
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result['ok'] else 1)
