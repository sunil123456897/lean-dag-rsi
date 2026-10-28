-- Lean-DAG-RSI v4 Static Formal Specification (Zero sorryAx, Exact Int/Rat)
namespace LeanDAGCert

def absInt (x : Int) : Int := if x < 0 then -x else x

def unitTriangleNormAreaScaled (p1 p2 p3 : Int × Int) : Int :=
  absInt (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

end LeanDAGCert
