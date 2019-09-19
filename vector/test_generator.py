import random

MAX_VECTOR_SIZE = 2 ** 25
PRECISION = 10
TEST_DIR = 'test/'


def write_test_input(test_name, size, v1, v2):
    with open(TEST_DIR + test_name + '.quest', 'w') as test:
        test.write(str(size) + '\n')
        test.write(' '.join(map(str, v1)) + '\n')
        test.write(' '.join(map(str, v2)) + '\n')


def write_test_output(test_name, v):
    with open(TEST_DIR + test_name + '.answ', 'w') as test:
        test.write(' '.join(['{0:.10e}'.format(e) for e in v]))


def main():

    # =======================

    test_name = 'basic_test'

    a, b, c = [], [], []
    size = 2 ** 10
    for _ in range(size):
        x = round(random.uniform(0.0, 666.0),
                  random.randint(0, PRECISION + 1))
        y = round(random.uniform(0.0, 666.0),
                  random.randint(0, PRECISION + 1))

        a.append(x)
        b.append(y)
        c.append(x - y)

    write_test_input(test_name=test_name, size=size, v1=a, v2=b)
    write_test_output(test_name=test_name, v=c)

    # =======================

    test_name = 'big_test'

    a, b, c = [], [], []
    size = MAX_VECTOR_SIZE - 1
    for _ in range(size):
        x = round(random.uniform(0.0, 666.0),
                  random.randint(0, PRECISION + 1))
        y = round(random.uniform(0.0, 666.0),
                  random.randint(0, PRECISION + 1))

        a.append(x)
        b.append(y)
        c.append(x - y)

    write_test_input(test_name=test_name, size=size, v1=a, v2=b)
    write_test_output(test_name=test_name, v=c)

    # =======================

main()

