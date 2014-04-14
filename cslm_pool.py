import sys

sys.stdout = open('cslm_pool.log', 'w')
sys.stderr = open('cslm_pool.log', 'w')
print "Test"

def run_cslm():
    print "Method called"
