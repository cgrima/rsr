#!/bin/bash

coverage3 run ./example.py
coverage3 run -a test_incircles.py
coverage3 run -a test_along.py
