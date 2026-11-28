"""Replay the case 39-5 canvas drag across VC3D correction and export builds."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import time
import uuid

import numpy as np

from verify_anchor_fix import VOLUME, load_mesh


class BridgeError(Exception):
    pass


class Bridge:
    def __init__(self, connection):
        self.stream = connection.makefile('rwb')
        self.sequence = 0

    def call(self, method, **params):
        self.sequence += 1
        message = dict(jsonrpc='2.0', id=self.sequence, method=method, params=params)
        self.stream.write((json.dumps(message)+'\n').encode())
        self.stream.flush()
        for line in self.stream:
            result = json.loads(line)
            if result.get('id') == self.sequence:
                if 'error' in result:
                    raise BridgeError(result['error'])
                return result['result']
        raise BridgeError('VC3D closed the bridge')

    def wait(self, job):
        deadline = time.monotonic()+90
        while time.monotonic() < deadline:
            info = self.call('job.status', jobId=job)
            if info['state'] not in ('queued', 'running', 'pending'):
                return info
            time.sleep(.25)
        raise TimeoutError(f'VC3D job {job} did not finish')

    def export(self, uuid, dest):
        info = self.call('objects.export', uuid=uuid, path=str(dest))
        if info.get('jobId'):
            info = self.wait(info['jobId'])
            if info['state'] != 'succeeded':
                raise BridgeError(dict(state=info['state'], message=info.get('message')))
        if not (dest/'x.tif').is_file():
            raise BridgeError('Export did not create mesh files')


WINDOW_ARGS = []  # e.g. ['-qwindowgeometry', '2300x1050+0+0'] (Qt's own option; no window manager under Xvfb)


@contextmanager
def start_gui(binary, workspace, tag, environment):
    name = f'edge-repair-39-5-{os.getpid()}-{tag}'
    with (workspace/f'{tag}.log').open('w', encoding='utf-8') as log:
        p = subprocess.Popen([str(binary), *WINDOW_ARGS, '--agent-bridge-name', name], cwd=workspace,
                             env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(30)
                endpoint = f'/tmp/{name}'
                deadline = time.monotonic()+45
                while True:
                    try:
                        connection.connect(endpoint)
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        if p.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError(f'VC3D did not start. See {workspace/tag}.log')
                        time.sleep(.2)
                bridge = Bridge(connection)
                try:
                    yield bridge
                finally:
                    bridge.stream.close()
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


def activate(gui, segment_id, timeout=30):
    """Activate a segment, waiting (bounded) for an import to reach the editor's surface list."""
    deadline = time.monotonic()+timeout
    while True:
        try:
            if gui.call('segments.activate', segmentId=segment_id)['activated']:
                return
            last = 'activated=false'
        except BridgeError as exc:
            last = exc.args[0]
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Segment {segment_id} did not activate within {timeout} s: {last}')
        time.sleep(.25)


def geom(mesh, baseline, row, col, target):
    arr = load_mesh(mesh)
    delta = np.linalg.norm(arr-baseline, axis=-1)
    return dict(anchor_error_vox=float(np.linalg.norm(arr[row,col]-target)),
                anchor_motion_vox=float(delta[row,col]), seed_changed=int(np.count_nonzero(delta[:, :24])),
                seed_max_motion_vox=float(delta[:, :24].max()), vertices_changed=int(np.count_nonzero(delta)))


def open_scan(gui, project, volume, voxel_size):
    opened = gui.call('volume.open', path=str(project))
    state = gui.call('state.get')
    locations = [entry['location'] for entry in state['project']['entries'] if entry['kind']=='volume']
    if (not opened['opened'] or locations != [volume] or state['volume']['id'] != opened['volumeId']
            or abs(state['volume']['voxelSize'] - voxel_size) > .001):
        raise RuntimeError(f'Unexpected volume: {opened}, {locations}, {state.get("volume")}')


def case_environment(args, variant, workspace):
    library = args.variants/variant/'libvc_tracer.so'
    build = args.cli.parent.parent
    folders = (library.parent, build/'core', build/'reader', args.libs)
    environment = {**os.environ, 'LD_LIBRARY_PATH': ':'.join(map(str, folders)),
                   'QT_QPA_PLATFORM':os.environ.get('QT_QPA_PLATFORM', 'offscreen'),
                   'OMP_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':'-1',
                   'OPENBLAS_NUM_THREADS':'1', 'VC_GROWPATCH_RNG_SEED':'3915'}
    for key in ('XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME','XDG_STATE_HOME'):
        isolated = workspace/key.lower()
        isolated.mkdir()
        environment[key] = str(isolated)
    for binary in (args.gui, args.cli):
        linker = subprocess.check_output(['ldd', str(binary)], env=environment, text=True)
        entry, = [line for line in linker.splitlines() if 'libvc_tracer.so =>' in line]
        if str(library) not in entry:
            raise RuntimeError(f'{binary.name} did not resolve {library}')
    return environment


def store_history(project, mesh_id):
    store = project.with_suffix('.vcobj')
    owner = bytes.fromhex(mesh_id.replace('-', ''))
    with sqlite3.connect(store.as_uri()+'?mode=ro', uri=True) as db:
        geometry = db.execute('''SELECT hex(uuid), hex(parent_uuid), hex(content_hash), object_version
                                 FROM geometry_revisions WHERE object_uuid = ? ORDER BY object_version''', (owner,)).fetchall()
        collections = db.execute('''SELECT hex(e.uuid), e.version, hex(c.uuid), c.version, c.meta
                                    FROM edges e JOIN objects c ON c.uuid = CASE WHEN e.a_uuid = ? THEN e.b_uuid ELSE e.a_uuid END
                                    WHERE (e.a_uuid = ? OR e.b_uuid = ?) AND e.state = 0 AND c.state = 0
                                      AND json_extract(e.meta, '$.role') IN ('corrections', 'growth_annotation')
                                    ORDER BY e.uuid''', (owner, owner, owner)).fetchall()
        version = db.execute('SELECT version FROM objects WHERE uuid = ?', (owner,)).fetchone()[0]
    history = dict(geometry=geometry, owner_version=version, corrections=[], growth_annotations=[])
    for edge_id, edge_version, collection_id, collection_version, meta in collections:
        meta = json.loads(meta)
        record = dict(edge_id=edge_id, edge_version=edge_version, collection_id=collection_id,
                      collection_version=collection_version)
        if meta['role'] == 'corrections':
            history['corrections'].append(dict(record, anchor=meta['source']['anchor2d']))
        else:
            history['growth_annotations'].append(dict(record, provenance=meta['source']['growth_annotation']))
    return history


def save_result(gui):
    save = gui.call('segmentation.save')
    return gui.wait(save['jobId']) if save.get('jobId') else save


def export_collection(gui, collection_id, path):
    result = gui.call('objects.export', uuid=collection_id, path=str(path))
    result = gui.wait(result['jobId'])
    if result['state'] != 'succeeded':
        raise BridgeError(dict(state=result['state'], message=result.get('message')))
    collection, = json.loads(path.read_text(encoding='utf-8'))['collections'].values()
    assert collection['uuid'] == collection_id
    return collection


def test_gui(args, workspace, environment, source, mesh_id, baseline, row, col, target, switch_segment=False):
    project = workspace/'test.vcproj'
    at = dict(zip('xyz', baseline[row,col].tolist()))
    destination = dict(zip('xyz', target.tolist()))
    result = {}
    with start_gui(args.gui, workspace, 'edit', environment) as gui:
        gui.call('project.create', path=str(project), name='VC3D correction regression', volume=args.volume)
        open_scan(gui, project, args.volume, args.voxel_size)
        task = gui.call('objects.import', kind='surface', path=str(source), bind={'scale':1})
        assert gui.wait(task['jobId'])['state'] == 'succeeded'
        if args.point_fix_gui:
            name = 'Ordinary volume annotation'
            point = gui.call('points.commit', collection=name, points=[at])
            volume_collection = point['uuid']
            normal = export_collection(gui, volume_collection, workspace/'ordinary_before.json')
            assert len(normal['points']) == 1
        if switch_segment:
            other_mesh = workspace/'secondary_input.tifxyz'
            shutil.copytree(source, other_mesh)
            other_id = str(uuid.uuid4())
            meta_path = other_mesh/'meta.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            meta.update(uuid=other_id, name='Secondary surface for switch test')
            meta_path.write_text(json.dumps(meta), encoding='utf-8')
            task = gui.call('objects.import', kind='surface', path=str(other_mesh), bind={'scale': 1})
            assert gui.wait(task['jobId'])['state'] == 'succeeded'
        activate(gui, mesh_id)
        pristine = workspace/'pristine.tifxyz'
        gui.export(mesh_id, pristine)
        result['unmodified_export_equal'] = bool(np.array_equal(load_mesh(pristine), baseline))
        assert result['unmodified_export_equal']

        state = gui.call('state.get')
        views = {v['surfName']:v for v in state['viewers']}
        flat = views['segmentation']['viewerId']
        xy = views['xy plane']['viewerId']
        gui.call('viewer.center_on_point', viewer=flat, point=at)
        gui.call('canvas.click', viewer=flat, position=at, modifiers=['ctrl'])
        state = gui.call('state.get')
        focus = state['focusPoi']['position']
        assert all(abs(focus[k]-at[k])<.1 for k in 'xyz')
        zoom = next(v['scale'] for v in state['viewers'] if v['viewerId']==xy)
        gui.call('viewer.zoom', viewer=xy, factor=2/zoom)
        gui.call('viewer.center_on_point', viewer=xy, point=at)
        gui.call('segmentation.enable_editing', enabled=True)
        gui.call('segmentation.corrections.set_point_mode', active=True)
        result['drag'] = gui.call('canvas.drag', viewer=xy, button='left', steps=8,
                                  **{'from': at, 'to': destination})
        state = gui.call('job.status', source='growth')
        result['solver'] = gui.wait(state['jobId'])
        if args.capture:
            # Growth invalidates visible CT tiles. Allow their asynchronous reload to settle.
            time.sleep(5)
            screenshot = workspace / 'after_drag.png'
            result['screenshot'] = gui.call('screenshot.capture', target=xy, filePath=str(screenshot))
            assert screenshot.is_file()
        try:
            result['save'] = save_result(gui)
        except BridgeError as exc:
            result['save_error'] = exc.args[0]
        if args.capture_save:
            # Whole window, so the editor's own save status is visible next to the CT view.
            time.sleep(1)
            shot = workspace / 'after_save_window.png'
            result['save_screenshot'] = gui.call('screenshot.capture', target='window', filePath=str(shot))
            assert shot.is_file()
        try:
            corrected = workspace/'gui_export.tifxyz'
            gui.export(mesh_id, corrected)
            result['export_geometry'] = geom(corrected, baseline, row, col, target)
        except BridgeError as exc:
            result['export_error'] = exc.args[0]
        if args.save_fix_gui:
            result['history_after_export'] = store_history(project, mesh_id)
            result['repeat_save'] = save_result(gui)
            result['history_after_repeat_save'] = store_history(project, mesh_id)
        if args.point_fix_gui:
            correction, = result['history_after_repeat_save']['corrections']
            correction_id = str(uuid.UUID(correction['collection_id']))
            # Let the volume-point timer flush any notifications from the correction.
            time.sleep(.75)
            try:
                points = export_collection(gui, correction_id, workspace/'correction_export.json')
                result['correction_export'] = points
            except BridgeError as exc:
                result['correction_export_error'] = exc.args[0]
        if switch_segment:
            assert gui.call('segments.activate', segmentId=other_id)['activated']
            try:
                points = export_collection(gui, volume_collection, workspace/'ordinary_after_switch.json')
                result['ordinary_after_switch_equal'] = normal == points
            except BridgeError as exc:
                result['ordinary_after_switch_error'] = exc.args[0]

    with start_gui(args.gui, workspace, 'reopen', environment) as gui:
        open_scan(gui, project, args.volume, args.voxel_size)
        reopened = workspace/'reopened.tifxyz'
        try:
            gui.export(mesh_id, reopened)
            result['reopened_geometry'] = geom(reopened, baseline, row, col, target)
            if 'export_geometry' in result:
                result['export_matches_reopened'] = bool(np.array_equal(load_mesh(corrected),load_mesh(reopened)))
        except BridgeError as exc:
            result['reopen_error'] = exc.args[0]
        if args.save_fix_gui:
            result['history_after_reopen'] = store_history(project, mesh_id)
        if args.point_fix_gui:
            reopened_points = export_collection(gui, volume_collection, workspace/'ordinary_reopened.json')
            result['ordinary_annotation_equal'] = normal == reopened_points
        if switch_segment:
            growth, = result['history_after_reopen']['growth_annotations']
            growth_id = str(uuid.UUID(growth['collection_id']))
            result['growth_annotation_export'] = export_collection(gui, growth_id, workspace/'growth_reopened.json')
    return result


def test_unanchored(args, workspace, environment, source, corrections):
    corrections['collections']['1'].pop('anchor2d',None)
    config = workspace/'unanchored.json'
    config.write_text(json.dumps(corrections, indent=2), encoding='utf-8')
    out = workspace/'unanchored.tifxyz'
    command = [str(args.cli),'--volume',args.volume,'--target-dir',str(out),'--params',str(args.fixture/'replay_params.json'),
               '--resume',str(source),'--correct',str(config),'--segment-name','unanchored','--rng-seed','3915']
    with (workspace/'unanchored.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=150,check=True)
    return load_mesh(out)


def verify_save_comparison(output, results, row, col, target):
    before = results['before_save_fix']
    assert before['save']['state'] == 'failed' and before['export_error']['code'] == -32005
    for arm, after in results.items():
        if arm == 'before_save_fix':
            continue
        assert after['save']['state'] == 'succeeded'
        repeat = after['repeat_save']
        assert repeat['state'] in {'idle', 'succeeded'}
        if repeat['state'] == 'idle':
            assert not (repeat['pending'] or repeat['saveInProgress'] or repeat['dirtyAfterSave'])
        assert after['export_matches_reopened'] and after['reopened_geometry'] == before['reopened_geometry']
        assert np.array_equal(load_mesh(output/'before_save_fix/reopened.tifxyz'),
                              load_mesh(output/arm/'gui_export.tifxyz'))
        history = after['history_after_export']
        assert history == after['history_after_repeat_save']
        assert len(history['geometry']) >= 2
        assert history['geometry'][-1][1] == history['geometry'][-2][0]
        correction, = history['corrections']
        assert correction['anchor'] == [col+2, row+2]
        reopened = after['history_after_reopen']
        if 'growth_annotation_export' in after:
            # The editor retires the applied, in-progress correction set when leaving the segment.
            # The immutable growth annotation and both surface revisions are the lasting record.
            assert reopened['geometry'] == history['geometry'] and not reopened['corrections']
            assert reopened['growth_annotations'] == history['growth_annotations']
            growth, = reopened['growth_annotations']
            provenance = growth['provenance']
            assert provenance['before_geometry'] == str(uuid.UUID(reopened['geometry'][-2][0]))
            assert provenance['after_geometry'] == str(uuid.UUID(reopened['geometry'][-1][0]))
            annotation = after['growth_annotation_export']
            assert annotation['growth_annotation'] == provenance
            point, = annotation['points'].values()
            assert np.allclose(point['p'], target, rtol=0, atol=.01)
        else:
            assert reopened == history
    print('Surface save and export passed with preserved geometry and correction history.', flush=True)

    if 'after_point_fix' in results:
        after = results['after_point_fix']
        points = after['correction_export']
        assert points['anchor2d'] == [col+2, row+2]
        point, = points['points'].values()
        assert np.allclose(point['p'], target, rtol=0, atol=.01)
        assert after['ordinary_after_switch_equal']
        assert all(result['ordinary_annotation_equal'] for result in results.values())
        warning = 'Point collections: could not save to the object store:'
        assert warning not in (output/'after_point_fix/edit.log').read_text(encoding='utf-8')
        control = results['after_save_fix']
        reproduced = (control.get('correction_export_error') or control.get('ordinary_after_switch_error')
                      or warning in (output/'after_save_fix/edit.log').read_text(encoding='utf-8'))
        status = 'reproduced' if reproduced else 'inconclusive (control did not reproduce the conflict)'
        print(f'Point collection exports passed. Deliberate switch regression: {status}.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('gui','cli','variants','libs','fixture','output'):
        p.add_argument(arg,type=lambda name: Path(name).resolve())
    p.add_argument('--volume', default=VOLUME)
    p.add_argument('--capture', action='store_true', help='capture the actual VC3D window after each drag')
    p.add_argument('--capture-save', action='store_true', help='capture the whole VC3D window after each save')
    p.add_argument('--save-fix-gui', type=lambda name: Path(name).resolve(),
                   help='compare save behavior with the patched anchor library fixed in both arms')
    p.add_argument('--point-fix-gui', type=lambda name: Path(name).resolve(),
                   help='include a third arm with the volume-point sync fix')
    p.add_argument('--window-geometry', help='initial VC3D window geometry, e.g. 2300x1050+0+0')
    args = p.parse_args()
    if args.window_geometry:
        WINDOW_ARGS[:] = ['-qwindowgeometry', args.window_geometry]
    if args.point_fix_gui and not args.save_fix_gui:
        p.error('--point-fix-gui requires --save-fix-gui')
    args.voxel_size = json.loads((args.fixture/'replay_params.json').read_text(encoding='utf-8'))['voxelsize']
    args.output.mkdir(exist_ok=False)
    source = args.fixture/'untrimmed.tifxyz'
    mesh_id = json.loads((source/'meta.json').read_text(encoding='utf-8'))['uuid']
    correction = (args.fixture/'correction.json').read_text(encoding='utf-8')
    values = json.loads(correction)['collections']['1']
    col,row = values['anchor2d']
    target = np.array(values['points']['1']['p'])
    baseline = load_mesh(source)
    if args.save_fix_gui:
        results = {}
        arms = [('before_save_fix', args.gui), ('after_save_fix', args.save_fix_gui)]
        if args.point_fix_gui:
            arms.append(('after_point_fix', args.point_fix_gui))
        for arm, binary in arms:
            args.gui = binary
            workspace = args.output/arm
            workspace.mkdir()
            env = case_environment(args, 'patched', workspace)
            results[arm] = test_gui(args, workspace, env, source, mesh_id, baseline, row, col, target,
                                    switch_segment=bool(args.point_fix_gui and arm != 'before_save_fix'))
            (workspace/'gui_result.json').write_text(json.dumps(results[arm], indent=2), encoding='utf-8')
            print(arm, 'save:', results[arm]['save'].get('state'), 'export:', results[arm].get('export_error'),
                  'repeat save:', results[arm]['repeat_save'].get('state'),
                  'correction export:', results[arm].get('correction_export_error', 'no error'),
                  'annotation after switch:', results[arm].get('ordinary_after_switch_error', 'no error'), flush=True)
        (args.output/'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        verify_save_comparison(args.output, results, row, col, target)
        return
    results, unanchored, environments = {}, {}, {}
    for arm in ('stock','patched'):
        workspace = args.output/arm
        workspace.mkdir()
        env = case_environment(args,arm,workspace)
        environments[arm] = env
        print('START',arm,flush=True)
        result = test_gui(args,workspace,env,source,mesh_id,baseline,row,col,target)
        (workspace/'gui_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        results[arm] = result
        print('DONE',arm,'export error:',result.get('export_error'),'reopened:',result.get('reopened_geometry'),flush=True)
    for arm, env in environments.items():
        unanchored[arm] = test_unanchored(args,args.output/arm,env,source,json.loads(correction))
    results['unanchored'] = dict(full_mesh_equal=bool(np.array_equal(unanchored['stock'],unanchored['patched'])),
                                  moved_vertices=int(np.count_nonzero(np.linalg.norm(unanchored['stock']-baseline,axis=-1))))
    (args.output/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Unanchored regression:',results['unanchored'],flush=True)


if __name__=='__main__':
    main()
