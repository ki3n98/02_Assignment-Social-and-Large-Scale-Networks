graph [
  comment "Balanced signed graph with two color groups (red/blue).
  Positive edges within groups, negative edges between groups.
  Use with: --plot P, --verify_homophily, --verify_balanced_graph"
  node [
    id 0
    label "0"
    color "red"
  ]
  node [
    id 1
    label "1"
    color "red"
  ]
  node [
    id 2
    label "2"
    color "red"
  ]
  node [
    id 3
    label "3"
    color "blue"
  ]
  node [
    id 4
    label "4"
    color "blue"
  ]
  node [
    id 5
    label "5"
    color "blue"
  ]
  edge [
    source 0
    target 1
    sign 1
  ]
  edge [
    source 1
    target 2
    sign 1
  ]
  edge [
    source 0
    target 2
    sign 1
  ]
  edge [
    source 3
    target 4
    sign 1
  ]
  edge [
    source 4
    target 5
    sign 1
  ]
  edge [
    source 3
    target 5
    sign 1
  ]
  edge [
    source 0
    target 3
    sign -1
  ]
  edge [
    source 1
    target 4
    sign -1
  ]
  edge [
    source 2
    target 5
    sign -1
  ]
]
