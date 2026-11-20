-- Lean-DAG-RSI v4 Static Formal Specification (Zero sorryAx, Exact Int/Rat)
namespace LeanDAGCert

def absInt (x : Int) : Int := if x < 0 then -x else x

def unitTriangleNormAreaScaled (p1 p2 p3 : Int × Int) : Int :=
  absInt (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

def inUnitTriangleScaled (scale : Int) (p : Int × Int) : Bool :=
  decide (0 <= p.1 ∧ 0 <= p.2 ∧ p.1 + p.2 <= scale)

def allInUnitTriangleScaled (scale : Int) (pts : List (Int × Int)) : Bool :=
  pts.all (inUnitTriangleScaled scale)

def pairWithThird (p1 p2 : Int × Int) (rest : List (Int × Int)) (boundNum boundDen scaleSq : Int) : Bool :=
  match rest with
  | [] => true
  | p3 :: tl =>
      decide (unitTriangleNormAreaScaled p1 p2 p3 * boundDen >= boundNum * scaleSq) &&
      pairWithThird p1 p2 tl boundNum boundDen scaleSq

def firstWithPairs (p1 : Int × Int) (rest : List (Int × Int)) (boundNum boundDen scaleSq : Int) : Bool :=
  match rest with
  | [] => true
  | p2 :: tl =>
      pairWithThird p1 p2 tl boundNum boundDen scaleSq &&
      firstWithPairs p1 tl boundNum boundDen scaleSq

def allTriplesMeetBound (pts : List (Int × Int)) (scale boundNum boundDen : Int) : Bool :=
  let scaleSq := scale * scale
  let rec go : List (Int × Int) -> Bool
    | [] => true
    | p1 :: tl => firstWithPairs p1 tl boundNum boundDen scaleSq && go tl
  go pts

def heilbronnCertificateValid (pts : List (Int × Int)) (nExpected : Nat) (scale boundNum boundDen : Int) : Bool :=
  decide (pts.length == nExpected) &&
  allInUnitTriangleScaled scale pts &&
  allTriplesMeetBound pts scale boundNum boundDen

end LeanDAGCert
