"""Generate a deterministic audiovisual fixture; requires Pillow, numpy, ffmpeg, espeak-ng."""
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
SR, FPS, DURATION = 16000, 10, 180
FONT = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
font = ImageFont.truetype(FONT, 30)
small = ImageFont.truetype(FONT, 22)
large = ImageFont.truetype(FONT, 48)
latin = {size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size) for size in (22,30,48)}

def label(draw, xy, text, font, fill, anchor=None):
    if anchor:
        draw.text(xy, text, font=latin[font.size], fill=fill, anchor=anchor)
        return
    x,y=xy
    for char in text:
        face=latin[font.size] if ord(char)<128 or char=='·' else font
        draw.text((x,y), char, font=face, fill=fill)
        x+=draw.textlength(char,font=face)
colors = [('红色', '#ee5555'), ('蓝色', '#4488ff'), ('绿色', '#44cc88'), ('黄色', '#ffcc44')]
shapes = ['圆形', '正方形', '三角形']
audio = np.zeros(SR * DURATION, dtype=np.int16)
events = []
scenes = []
with tempfile.TemporaryDirectory() as tmp:
    for i in range(12):
        start = i * 15
        color, rgb = colors[i % 4]
        shape = shapes[i % 3]
        scenes.append(dict(start=start, end=start+15, scene=i+1, color=color, shape=shape,
                           number=i+1, direction='向右' if i % 2 == 0 else '向左'))
        prompts = [(start+2, '请说出现在画面中图形的颜色和形状。', f'{color}{shape}')]
        if i % 3 == 0:
            prompts.append((start+9, '图形正在向哪个方向移动？', scenes[-1]['direction']))
        elif i % 3 == 1:
            prompts.extend([(start+8, '请详细描述现在的画面。', f'{color}{shape}，编号{i+1}'),
                            (start+11, '等一下，只说图形里面的数字。', str(i+1))])
        else:
            prompts.append((start+9, '和上一段相比，图形颜色变了吗？只说现在的颜色。', color))
        for j, (at, text, expected) in enumerate(prompts):
            raw, wav = Path(tmp)/'raw.wav', Path(tmp)/'voice.wav'
            subprocess.run(['espeak-ng', '-v', 'cmn', '-s', '195', '-w', str(raw), text], check=True, capture_output=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-ar', str(SR), '-ac', '1', str(wav)], check=True)
            with wave.open(str(wav)) as f:
                samples = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').copy()
            # Bound each utterance before the next prompt / scene; retain the whole utterance by resampling.
            limit = int(((prompts[j+1][0] if j+1 < len(prompts) else start+14.5)-at-0.15)*SR)
            if len(samples) > limit:
                samples = np.interp(np.linspace(0,len(samples)-1,limit),np.arange(len(samples)),samples).astype(np.int16)
            offset = int(at*SR)
            audio[offset:offset+len(samples)] = samples
            events.append(dict(start=at,end=at+len(samples)/SR,text=text,expected=expected,
                               purpose='尝试打断；是否重叠取决于模型响应时延' if '等一下' in text else '当前画面理解'))
    with wave.open(str(OUT/'input.wav'), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(audio.tobytes())
    cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r',str(FPS),
           '-i','pipe:0','-i',str(OUT/'input.wav'),'-c:v','libx264','-preset','fast','-crf','23',
           '-pix_fmt','yuv420p','-c:a','aac','-b:a','64k','-movflags','+faststart','-t',str(DURATION),str(OUT/'duplex_sliding_window_180s.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in range(DURATION*FPS):
        t = frame/FPS; i = int(t//15); local=t%15
        color,rgb=colors[i%4]; shape=shapes[i%3]
        im=Image.new('RGB',(960,540),'#101928'); d=ImageDraw.Draw(im)
        label(d,(30,20),'全双工 / 滑动窗口测试',font=font,fill='white')
        label(d,(30,68),f'时间 {t:05.1f}s / 180s     场景 {i+1:02d} / 12',font=small,fill='#bbc8db')
        d.line((90,365,870,365),fill='#526078',width=3)
        x=int(145+670*(local/15 if i%2==0 else 1-local/15)); y=245
        if shape=='圆形': d.ellipse((x-70,y-70,x+70,y+70),fill=rgb)
        elif shape=='正方形': d.rectangle((x-70,y-70,x+70,y+70),fill=rgb)
        else: d.polygon([(x,y-85),(x-85,y+70),(x+85,y+70)],fill=rgb)
        label(d,(x,y),str(i+1),font=large,fill='#101928',anchor='mm')
        active=next((e for e in events if e['start']<=t<e['end']),None)
        label(d,(30,405),'用户说话中' if active else '用户静音 · 留给模型回答',font=small,fill='#66ddbb' if active else '#bbc8db')
        # Prompts intentionally absent from frames: audio comprehension cannot be bypassed with OCR.
        segment=audio[int(t*SR):int((t+0.1)*SR)]
        amplitude=float(np.abs(segment.astype(float)).mean())/32768 if len(segment) else 0
        d.rectangle((30,452,30+int(min(1,amplitude*7)*900),466),fill='#66ddbb')
        d.rectangle((30,505,30+int(t/DURATION*900),510),fill='#4488ff')
        proc.stdin.write(im.tobytes())
        if frame in (0,450,900,1350): im.save(OUT/f'preview_{int(t):03d}s.jpg')
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
(OUT/'timeline.json').write_text(json.dumps(dict(duration=DURATION,fps=FPS,audio_sample_rate=SR,scenes=scenes,utterances=events),ensure_ascii=False,indent=2))
print(OUT/'duplex_sliding_window_180s.mp4')
