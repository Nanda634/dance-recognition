"""Reference-based dance pose comparison for the existing MediaPipe 33x4 pipeline.
Reference poses are user-provided and should be instructor-verified for teaching.
"""
from pathlib import Path
import json
import re
import numpy as np
import cv2
from src.pose import get_pose

JOINTS = {
    'Left elbow': (11,13,15), 'Right elbow': (12,14,16),
    'Left shoulder': (13,11,23), 'Right shoulder': (14,12,24),
    'Left hip': (11,23,25), 'Right hip': (12,24,26),
    'Left knee': (23,25,27), 'Right knee': (24,26,28),
}
DANCE_FORMS = ['Bharatanatyam','Kuchipudi','Kathak','Odissi','Mohiniyattam','Kathakali', 'Manipuri','Sattriya']

def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_-]+','_',s.strip())[:65].strip('_') or 'pose'

def extract(image):
    feature, annotated = get_pose(image, draw=True, complexity=0)
    if feature is None:
        raise ValueError('No body pose detected. Use a clear, full-body photograph.')
    return np.asarray(feature, dtype=float).reshape(33,4), annotated

def angle(points, a,b,c):
    v1=points[a,:2]-points[b,:2]
    v2=points[c,:2]-points[b,:2]
    norm=np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm < 1e-8: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm,-1,1))))

def compare(student, reference, tolerance=15.0, min_visibility=0.45):
    student=np.asarray(student,dtype=float).reshape(33,4)
    reference=np.asarray(reference,dtype=float).reshape(33,4)
    rows=[]
    for name,(a,b,c) in JOINTS.items():
        if min(student[[a,b,c],3].min(), reference[[a,b,c],3].min()) < min_visibility:
            rows.append({'Joint':name,'Reference angle':None,'Student angle':None,'Difference':None,'Feedback':'Not reliably visible'})
            continue
        target=angle(reference,a,b,c); actual=angle(student,a,b,c)
        if target is None or actual is None: continue
        diff=actual-target
        if abs(diff)<=tolerance: feedback='Close to reference'
        elif diff>0: feedback='Bend more to match reference'
        else: feedback='Straighten more to match reference'
        rows.append({'Joint':name,'Reference angle':round(target,1),'Student angle':round(actual,1),'Difference':round(abs(diff),1),'Feedback':feedback})
    valid=[r for r in rows if r['Difference'] is not None]
    matched=sum(r['Difference']<=tolerance for r in valid)
    return {'rows':rows,'matched':matched,'measured':len(valid),'match_percent':round(100*matched/len(valid),1) if valid else None}

def reference_paths(root, dance):
    base=Path(root)/safe_name(dance)
    base.mkdir(parents=True,exist_ok=True)
    return base

def save_reference(root, dance, pose_name, image_bgr, notes=''):
    arr,annotated=extract(image_bgr)
    base=reference_paths(root,dance)
    stem=safe_name(pose_name)
    path=base/(stem+'.json')
    data={'dance':dance,'pose_name':pose_name,'notes':notes,'landmarks':arr.tolist(),
          'warning':'User-provided reference; not automatically instructor-verified'}
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    cv2.imwrite(str(base/(stem+'.jpg')),image_bgr)
    return path,annotated

def list_references(root,dance):
    return sorted(reference_paths(root,dance).glob('*.json'))

def load_reference(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    data['landmarks']=np.asarray(data['landmarks'],dtype=float).reshape(33,4)
    return data

def analyze_video(path,reference,tolerance=15,max_samples=24):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError('Could not open video')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples=[]; preview=None
    try:
        if count>0:
            positions=np.unique(np.linspace(0,max(count-1,0),min(max_samples,count)).astype(int))
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(pos))
                ok,frame=cap.read()
                if not ok: continue
                try:
                    arr,annotated=extract(frame)
                    result=compare(arr,reference,tolerance)
                    if result['match_percent'] is not None:
                        samples.append({'Frame':int(pos),'Match %':result['match_percent'], 'Result':result})
                        if preview is None: preview=annotated
                except ValueError: pass
        else:
            for pos in range(max_samples):
                ok,frame=cap.read()
                if not ok: break
                try:
                    arr,annotated=extract(frame)
                    result=compare(arr,reference,tolerance)
                    if result['match_percent'] is not None:
                        samples.append({'Frame':pos,'Match %':result['match_percent'],'Result':result})
                        if preview is None: preview=annotated
                except ValueError: pass
    finally: cap.release()
    return samples,preview