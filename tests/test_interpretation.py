import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.contracts import ContractError
from home_lab_voice.interpretation import interpret, local_endpoint, validate_proposal

TEXT='Show status for voice-service'
ACCEPT={'disposition':'accept','intent':'lab.status.read','arguments':{'service':'voice-service'},'evidence_quotes':[TEXT]}

class InterpretationTests(unittest.TestCase):
    def test_valid_proposal_is_data_not_execution(self):
        result=validate_proposal(json.dumps(ACCEPT),TEXT)
        self.assertEqual(result.arguments,{'service':'voice-service'})

    def test_wrong_transcript_evidence_rejected(self):
        with self.assertRaises(ContractError): validate_proposal(json.dumps(ACCEPT),'Other input')

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ContractError):
            validate_proposal('{"disposition":"accept","disposition":"abstain","intent":null,"arguments":{},"evidence_quotes":[]}',TEXT)

    def test_nonaccept_cannot_carry_action(self):
        for disposition in ('clarify','abstain'):
            with self.subTest(disposition=disposition), self.assertRaises(ContractError):
                validate_proposal(json.dumps(dict(ACCEPT,disposition=disposition)),TEXT)

    def test_malformed_and_out_of_scope_payloads(self):
        payloads=[[],None,dict(ACCEPT,extra=True),dict(ACCEPT,intent='shell.run'),
                  dict(ACCEPT,arguments={'service':'external'}),dict(ACCEPT,arguments={'service':['voice-service']}),
                  dict(ACCEPT,evidence_quotes=['']),dict(ACCEPT,evidence_quotes=[TEXT]*5),
                  dict(ACCEPT,disposition={'accept':True})]
        for payload in payloads:
            with self.subTest(payload=payload),self.assertRaises(ContractError):
                validate_proposal(json.dumps(payload),TEXT)

    def test_schema_does_not_prove_semantic_correctness(self):
        # A matching quotation can still support the wrong interpretation.
        negative='Do not show status for voice-service'
        candidate=dict(ACCEPT,evidence_quotes=[negative])
        self.assertEqual(validate_proposal(json.dumps(candidate),negative).disposition,'accept')

    def test_endpoint_rejects_remote_credentials_and_proxy_targets(self):
        for endpoint in ('https://example.com/v1','http://192.168.1.2:8000/v1',
                         'http://user:pass@127.0.0.1/v1','http://127.0.0.1/v1?target=remote',
                         'file:///tmp/model','http://127.0.0.1/v1#x'):
            with self.subTest(endpoint=endpoint),self.assertRaises(ContractError):local_endpoint(endpoint)
        self.assertEqual(local_endpoint('http://[::1]:8000/v1/'),'http://[::1]:8000/v1')

    def test_http_response_and_redirect_boundary(self):
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(body)
                if body['model']=='redirect':
                    self.send_response(307);self.send_header('Location','http://example.invalid/steal');self.end_headers();return
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps({'model':'fixture','choices':[{'finish_reason':'stop',
                    'message':{'content':json.dumps(ACCEPT)}}]}).encode())
        server=HTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            endpoint=f'http://127.0.0.1:{server.server_port}/v1'
            result=interpret(TEXT,endpoint=endpoint,model='fixture')
            self.assertTrue(result['valid']);self.assertEqual(len(seen),1)
            with self.assertRaises(ContractError) as error:interpret(TEXT,endpoint=endpoint,model='redirect')
            self.assertEqual(error.exception.code,'MODEL_REDIRECT_BLOCKED')
            self.assertEqual(len(seen),2)
        finally:server.shutdown();server.server_close();worker.join(timeout=2)

if __name__=='__main__':unittest.main()
