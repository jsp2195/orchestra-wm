"""Create measured acquisition/provenance and run existing-adapter QC."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_continuous.qc import quality_control,roadway_coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--archive-root',required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--documentation-commit',required=True)
    a=p.parse_args();root=Path(a.data_root);archives=Path(a.archive_root)
    audit=json.loads((root/'source_audit.json').read_text())
    manifest=json.loads((archives/'DRIVE_ARCHIVE_MANIFEST.json').read_text())
    selection=audit['selection']
    meta=dict(dataset='I-24 MOTION',release='I24MOTION_PUBLIC_v1.0',format='json-array',direction=-1,
              terms_accepted=True,corrected_timestamps=True,source_url='https://i24motion.org/access_data',
              x_origin_ft=selection['x_max'],session=a.session,recording_date=audit['source_recording_date'],
              start_unix_s=selection['start'],end_unix_s=selection['start']+selection['seconds'],
              files=[dict(path='westbound_selected.json',bytes=audit['selected_bytes'],sha256=audit['selected_sha256'])])
    with zipfile.ZipFile(archives/'11-21-2022.zip') as z:
        readme=z.read('11-21-2022/README_(INCEPTION).txt')
    write_immutable(root/'acquisition.json',meta)
    write_immutable(root/'provenance.json',dict(
        acquisition=meta,archive_hashes=[{k:m[k] for k in ('name','id','bytes','sha256','md5')} for m in manifest],
        nested_zip_sha256=audit['nested_zip_sha256'],source_member=audit['source_member'],
        nested_member_crc_verified=audit['nested_member_crc_verified'],
        bundled_readme_sha256=hashlib.sha256(readme).hexdigest(),
        documentation_url='https://github.com/I24-MOTION/I24M_documentation',documentation_commit=a.documentation_commit,
        selection=selection,
        authorization_basis='User expressly requested private Drive download and research use; bundled README and public documentation permit academic use under the cited data use agreement',
        release_basis='Bundled INCEPTION session/date listing and public v1.x schema documentation; no publisher signature supplied',
        timestamp_basis='Public v1.x documentation defines timestamp as synchronization-corrected Unix seconds',
        coordinate_basis='v1.x curvilinear feet; x increases EB; WB s=-(x-x_origin)*0.3048+length_m/2; y retained in metres',
        citation='Gloudemans et al. 2023, Transportation Research Part C 155, 104311.',
        scope='Fixed 10-minute regional subset selected before model fitting; all source bytes preserved in ZIPs'))
    result=quality_control(root)
    roadway_coverage(root)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='REGIONAL_SUBSECOND_SUPPORT_PASSED' else 2


if __name__=='__main__':raise SystemExit(main())
