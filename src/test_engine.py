
from engine import Value
def test_engine():
    x1 = Value(3)
    x2 = Value(6)
    v1 = x1**2
    v2 = x1 * x2
    out = v1 -v2
    out.backward()
    print(x1)
    print(x2)

    
if __name__ == "__main__":
    test_engine()