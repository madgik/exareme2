#!/usr/bin/env bash
celery -A exareme2.worker.utils.celery_app worker -l INFO --pool eventlet
