"""A durable, local-only diagnostic project with real SQLite work queues.

The workspace belongs to the current OS user. It is not a multi-user security
boundary: anyone who can edit these files can change their contents. No shell,
remote target, arbitrary path, or external side effect is exposed to a model.
"""
from contextlib import contextmanager, closing
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
import time

from .contracts import ContractError

SERVICES=('voice-service','media-indexer')
DIAGNOSTICS=('healthcheck','queue_snapshot')
KEY=re.compile(r'[a-zA-Z0-9_-]{12,80}\Z')

def _fail(code,message):raise ContractError(code,message)
def _encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'))

class LocalLab:
    """Own only a new explicitly selected workspace and two bounded queue tables."""
    def __init__(self,root:Path):
        self.root=Path(root).absolute()
        self.database=self.root/'lab.sqlite'
        self._check_path()

    def _check_path(self):
        for path in (self.root,*self.root.parents):
            if path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction()):
                _fail('LAB_LINK_REJECTED','The workspace must not pass through a symbolic link or junction.')
        if self.database.is_symlink():_fail('LAB_LINK_REJECTED','The database must not be a symbolic link.')

    def initialize(self):
        self._check_path()
        # Refuse an existing directory, including an empty one, to avoid adopting unrelated files.
        self.root.mkdir(parents=False,mode=0o700,exist_ok=False)
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript('''
              CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
              CREATE TABLE jobs(id TEXT PRIMARY KEY,service TEXT NOT NULL,state TEXT NOT NULL);
              CREATE TABLE requests(id TEXT PRIMARY KEY,idem TEXT UNIQUE NOT NULL,service TEXT NOT NULL,
                diagnostic TEXT NOT NULL,expires REAL NOT NULL,digest TEXT NOT NULL,state TEXT NOT NULL,result TEXT);
            ''')
            db.executemany('INSERT INTO meta VALUES (?,?)',[('schema','local-lab-v1'),('workspace_id',secrets.token_hex(16))])
            db.commit()
        return {'state':'initialized','services':list(SERVICES),'scope':'Two local queue tables; no operating-system service management.'}

    @contextmanager
    def _connection(self):
        self._check_path()
        if not self.database.is_file() or self.database.stat().st_size>16*1024*1024:
            _fail('LAB_DATABASE_UNAVAILABLE','Initialize a bounded local lab database first.')
        # mode=rw prevents sqlite silently creating a replacement after deletion.
        db=sqlite3.connect(self.database.resolve().as_uri()+'?mode=rw',uri=True,timeout=5,isolation_level=None)
        db.row_factory=sqlite3.Row
        try:
            if db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0]!='local-lab-v1':
                _fail('LAB_SCHEMA_INVALID','Unsupported lab schema.')
            yield db
        finally:db.close()

    def _service(self,service):
        if service not in SERVICES:_fail('LAB_TARGET_REJECTED','Unknown local lab target.')

    def status(self,service):
        self._service(service)
        with self._connection() as db:
            counts={r['state']:r['n'] for r in db.execute('SELECT state,count(*) AS n FROM jobs WHERE service=? GROUP BY state',(service,))}
        return {'service':service,'queue_counts':counts,'state':'local_database_readable'}

    def enqueue(self,service,job_id):
        """Explicit typed CLI operation; deliberately absent from model tools."""
        self._service(service)
        if not isinstance(job_id,str) or not KEY.fullmatch(job_id):_fail('LAB_JOB_INVALID','Use a 12 to 80 character job identifier.')
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                row=db.execute('SELECT service FROM jobs WHERE id=?',(job_id,)).fetchone()
                if row and row['service']!=service:_fail('LAB_JOB_CONFLICT','Job identifier already belongs to another target.')
                if not row:
                    if db.execute('SELECT count(*) FROM jobs').fetchone()[0]>=1000:_fail('LAB_QUEUE_FULL','The project queue is full.')
                    db.execute('INSERT INTO jobs VALUES (?,?,?)',(job_id,service,'queued'))
                db.commit()
            except BaseException:db.rollback();raise
        return {'job_id':job_id,'service':service,'state':'queued','already_present':bool(row)}

    def prepare(self,service,diagnostic,idempotency_key,*,now=None,ttl=300):
        self._service(service)
        if diagnostic not in DIAGNOSTICS:_fail('LAB_DIAGNOSTIC_REJECTED','Unknown diagnostic.')
        if not isinstance(idempotency_key,str) or not KEY.fullmatch(idempotency_key):_fail('LAB_KEY_INVALID','Use a 12 to 80 character idempotency key.')
        if not isinstance(ttl,int) or not 1<=ttl<=300:_fail('LAB_TTL_INVALID','Approval lasts 1 to 300 seconds.')
        now=time.time() if now is None else now
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                row=db.execute('SELECT * FROM requests WHERE idem=?',(idempotency_key,)).fetchone()
                if row:
                    if (row['service'],row['diagnostic'])!=(service,diagnostic):_fail('LAB_KEY_CONFLICT','Idempotency key belongs to another operation.')
                    result=dict(row);result.pop('result');db.commit();return result
                if db.execute('SELECT count(*) FROM requests').fetchone()[0]>=1000:_fail('LAB_REQUEST_LIMIT','Archive the workspace and initialize another after 1000 requests.')
                request_id=secrets.token_hex(16);expires=now+ttl
                workspace=db.execute("SELECT value FROM meta WHERE key='workspace_id'").fetchone()[0]
                digest=hashlib.sha256(_encoded({'workspace':workspace,'id':request_id,'service':service,
                    'diagnostic':diagnostic,'idempotency_key':idempotency_key,'expires':expires}).encode()).hexdigest()
                db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,NULL)',(request_id,idempotency_key,service,diagnostic,expires,digest,'pending'))
                result=dict(db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone());result.pop('result');db.commit();return result
            except BaseException:db.rollback();raise

    def confirm(self,request_id,digest,*,now=None):
        now=time.time() if now is None else now
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                row=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
                if not row or not isinstance(digest,str) or not secrets.compare_digest(row['digest'],digest):
                    _fail('LAB_APPROVAL_INVALID','Approval does not match the saved operation.')
                if row['state']=='done':db.commit();return {'replayed':True,**json.loads(row['result'])}
                if row['state']!='pending':_fail('LAB_APPROVAL_REVOKED','This request is not pending.')
                if now>=row['expires']:_fail('LAB_APPROVAL_EXPIRED','Approval expired; use a new request key and review again.')
                self._service(row['service'])
                if row['diagnostic']=='healthcheck':
                    result={'database_check':db.execute('PRAGMA quick_check').fetchone()[0],
                            'jobs':db.execute('SELECT count(*) FROM jobs WHERE service=?',(row['service'],)).fetchone()[0]}
                elif row['diagnostic']=='queue_snapshot':
                    result={'queue_counts':{r['state']:r['n'] for r in db.execute('SELECT state,count(*) AS n FROM jobs WHERE service=? GROUP BY state',(row['service'],))}}
                else:_fail('LAB_DIAGNOSTIC_REJECTED','Unknown saved diagnostic.')
                record={'request_id':request_id,'service':row['service'],'diagnostic':row['diagnostic'],'observed_at':now,'result':result}
                db.execute("UPDATE requests SET state='done',result=? WHERE id=?",(_encoded(record),request_id))
                db.commit();return {'replayed':False,**record}
            except BaseException:db.rollback();raise

    def revoke(self,request_id):
        with self._connection() as db:
            changed=db.execute("UPDATE requests SET state='revoked' WHERE id=? AND state='pending'",(request_id,)).rowcount
        return {'request_id':request_id,'revoked':bool(changed)}
