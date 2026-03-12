from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Banker's Algorithm with Deadlock Recovery
def is_safe(available, allocation, maximum):
    num_processes = len(allocation)
    num_resources = len(available)

    work = available.copy()
    finish = [False] * num_processes
    safe_sequence = []

    need = maximum - allocation

    while len(safe_sequence) < num_processes:
        progress = False
        for i in range(num_processes):
            if not finish[i] and np.all(need[i] <= work):
                work += allocation[i]
                finish[i] = True
                safe_sequence.append(i)
                progress = True
                break

        if not progress:
            break

    is_system_safe = len(safe_sequence) == num_processes
    return is_system_safe, safe_sequence

def recover_from_deadlock(available, allocation, maximum):
    safe, sequence = is_safe(available.copy(), allocation.copy(), maximum.copy())
    log = ""

    if safe:
        return True, sequence, "No recovery needed."

    # Deadlock exists, recovery needed
    num_processes = len(allocation)
    while not safe:
        idx = np.argmax(np.sum(allocation, axis=1))  # kill max allocated
        log += f"Process {idx} terminated. Resources released: {allocation[idx].tolist()}\n"
        available += allocation[idx]
        allocation[idx] = np.zeros_like(allocation[idx])
        maximum[idx] = np.zeros_like(maximum[idx])

        safe, sequence = is_safe(available.copy(), allocation.copy(), maximum.copy())

    return False, sequence, log

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        num_processes = int(request.form['num_processes'])
        num_resources = int(request.form['num_resources'])

        available = np.array(list(map(int, request.form['available'].strip().split())))

        allocation = np.array([
            list(map(int, row.strip().split()))
            for row in request.form['allocation'].strip().split('\n')
        ])

        maximum = np.array([
            list(map(int, row.strip().split()))
            for row in request.form['maximum'].strip().split('\n')
        ])

        deadlock_free, sequence, recovery_log = recover_from_deadlock(
            available, allocation, maximum
        )

        return render_template(
            'result.html', safe=deadlock_free, sequence=sequence, recovery_log=recovery_log
        )

    except Exception as e:
        return f"<h2>Error processing input:</h2><pre>{e}</pre>"

if __name__ == '__main__':
    app.run(debug=True)
