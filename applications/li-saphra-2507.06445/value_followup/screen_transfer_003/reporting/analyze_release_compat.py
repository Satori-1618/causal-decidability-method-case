"""Post-run schema compatibility audit; no neural calls or scientific changes.

The frozen analyzer incorrectly expects final_review.status == 'approved'.
The frozen release validator and runner require the top-level status 'approved'
and the nested review status 'PASS'. This wrapper validates that complete
release schema, then corrects exactly the nested status literal in memory.
Frozen source files, run bindings, statistics, thresholds and raw data stay
unchanged. The report records both source hashes and the executed source hash.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
FROZEN_ANALYZER_SHA256 = '20475a4de85f4863cdf0047962f2d7dd8c86655c1af9c9664f385a900c58c2a9'
FROZEN_RELEASE_VALIDATOR_SHA256 = '93eb0bd0cf4c12c19d504fc48c74aca7ba1a8e97e1a5e8f3938af61d74280fca'
OLD = "release['final_review']['status'] == 'approved'"
NEW = "release['final_review']['status'] == 'PASS'"
REASON = ('The frozen analyzer used the top-level release label for the nested '
          'review status; the frozen runner requires release.status=approved '
          'and final_review.status=PASS. Only this literal is corrected.')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_review_label(source):
    if source.count(OLD) != 1:
        raise ValueError('Expected exactly one frozen nested-review status check')
    return source.replace(OLD, NEW, 1)


def transformed_source(source):
    if sha(source) != FROZEN_ANALYZER_SHA256:
        raise ValueError('Frozen analyzer source hash differs; no automatic patch allowed')
    return replace_review_label(source.decode('utf-8')).encode('utf-8')


def release_validator():
    path = ROUND/'verify_plan.py'
    if sha(path.read_bytes()) != FROZEN_RELEASE_VALIDATOR_SHA256:
        raise ValueError('Frozen execution-release validator source hash differs')
    sys.path.insert(0, str(ROUND))
    spec = importlib.util.spec_from_file_location('screen_transfer_frozen_release_validator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_execution_release


def load_analyzer():
    path = ROUND/'analyze_transfer.py'
    source = path.read_bytes()
    executed = transformed_source(source)
    module = types.ModuleType('screen_transfer_release_compatible_analyzer')
    # Preserve original path resolution and original on-disk provenance hashes.
    # The additional compatibility record exposes the different executed bytes.
    module.__file__ = str(path)
    exec(compile(executed, str(path), 'exec'), module.__dict__)
    return module, source, executed


def audit(run, inputs):
    analyzer, source, executed = load_analyzer()
    plan = analyzer.load_json(ROUND/'plan.json')
    release = analyzer.load_json(ROUND/'EXECUTION_RELEASE.json')
    release_validator()(plan, release)
    result = analyzer.audit(run, inputs)
    result['analysis_compatibility'] = {
        'status': 'post_run_schema_correction',
        'reason': REASON,
        'frozen_analyzer_sha256': sha(source),
        'executed_analyzer_source_sha256': sha(executed),
        'wrapper_sha256': sha(Path(__file__).read_bytes()),
        'frozen_release_validator_sha256': FROZEN_RELEASE_VALIDATOR_SHA256,
        'source_change_count': 1,
        'replaced_text': OLD,
        'replacement_text': NEW,
        'scientific_rules_changed': False,
        'frozen_files_or_raw_records_modified': False,
        'neural_rerun_performed': False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.inputs)
    if args.check_report:
        analyzer, _, _ = load_analyzer()
        analyzer.old.compare(analyzer.load_json(args.check_report), result)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(encoded)
    print(encoded)


if __name__ == '__main__':
    main()
