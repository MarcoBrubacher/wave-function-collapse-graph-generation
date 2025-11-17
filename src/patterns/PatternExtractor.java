package patterns;

import helper.Graph;
import helper.Node;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Builds ego-network Patterns for each node in a graph up to a given radius.
 * For each center node, performs a BFS to depth ≤ radius, captures node labels,
 * induced adjacency, layering by distance, and depths, then deduplicates
 * patterns using an explicit root-preserving, label-preserving graph
 * isomorphism check—incrementing frequency for duplicates.
 * <p>
 * This matches the mathematical definition of patterns as isomorphism classes
 * of rooted, vertex-labelled radius-r ego-networks.
 */
public final class PatternExtractor {
    // Prevent instantiation
    private PatternExtractor() {
    }

    /**
     * Extracts ego-network patterns for all vertices in the given graph at the
     * specified radius. Patterns that are identical up to root-preserving,
     * label-preserving graph isomorphism are merged, and their frequencies
     * are aggregated.
     *
     * For each vertex in the input graph this method:
     * - builds its radius-r ego-network as a rooted, vertex-labelled subgraph,
     * - checks whether an isomorphic pattern has already been seen,
     * - creates a new Pattern for a new isomorphism class, or
     * increments the frequency of the existing representative.
     *
     * The returned list therefore contains one Pattern object per
     * isomorphism class, each annotated with how many times it occurs.
     *
     * @param graph  the input graph; must not be null
     * @param radius maximum hop-distance (must be at least 1)
     * @return list of unique Pattern objects with aggregated frequencies
     * @throws IllegalArgumentException if graph is null or radius < 1
     */
    public static List<Pattern> extractPatterns(Graph graph, int radius) {
        Objects.requireNonNull(graph, "graph must not be null");
        if (radius < 1) {
            throw new IllegalArgumentException("radius must be ≥ 1");
        }

        // Store one representative Pattern per isomorphism class, in insertion order.
        List<Pattern> unique = new ArrayList<>();

        for (Node center : graph.getAllNodes()) {
            Pattern p = buildPattern(center, radius);
            Pattern existingIsomoprhClass = null;
            for (Pattern q : unique) {
                if (areRootedIsomorphic(p, q)) {
                    existingIsomoprhClass = q;
                    break;
                }
            }
            if (existingIsomoprhClass == null) {
                // New isomorphism class
                unique.add(p);
            } else {
                // Same isomorphism class, bump frequency
                existingIsomoprhClass.updateFrequency();
            }
        }
        return unique;
    }

    /**
     * Builds the radius-r ego-network pattern for a single centre node,
     * initialised with frequency 1.
     *
     * Starting from the centre node, this method performs a breadth-first search
     * up to the given radius and records:
     * - the distance of each reached node from the centre,
     * - the label of each node,
     * - the adjacency between all nodes within the radius, and
     * - the nodes at each exact distance (layers).
     *
     * This information is packaged into a Pattern whose ID and centre
     * label are taken from the centre node, whose radius is the given radius,
     * and whose frequency is initialised to 1.
     *
     * @param center the centre node; must not be null
     * @param radius maximum hop-distance (must be at least 1)
     * @return a new Pattern capturing the centre’s ego-network
     */
    private static Pattern buildPattern(Node center, int radius) {
        // 1) Compute BFS depths from center
        Map<Node, Integer> nodeDepths = computeDepths(center, radius);

        // 2) Map node IDs to labels and depths
        Map<Integer, Integer> labels = new LinkedHashMap<>();
        Map<Integer, Integer> depths = new LinkedHashMap<>();
        for (Map.Entry<Node, Integer> e : nodeDepths.entrySet()) {
            Node n = e.getKey();
            labels.put(n.getId(), n.getLabel());
            depths.put(n.getId(), e.getValue());
        }

        // 3) Build per-distance layers
        List<Set<Integer>> layers = computeLayers(depths, radius);

        // 4) Build induced adjacency among nodes within radius
        Set<Integer> subIds = labels.keySet();
        Map<Integer, List<Integer>> adjacency = new LinkedHashMap<>();
        for (Node n : nodeDepths.keySet()) {
            List<Integer> nbrs = n.getNeighbors().stream().map(Node::getId).filter(subIds::contains).collect(Collectors.toList());
            adjacency.put(n.getId(), nbrs);
        }

        // 5) Center node degree = number of neighbors at distance 1
        int centerNodeDegree = layers.get(0).size();

        // 6) Construct Pattern with frequency = 1
        return new Pattern(center.getId(), center.getLabel(), radius, labels, adjacency, layers, depths, 1, centerNodeDegree);
    }

    /**
     * Computes shortest-path distances from the given centre node up to the
     * specified radius using breadth-first search.
     * <p>
     * Only nodes that are reachable within the given radius are included.
     * The centre node has distance 0, its neighbours distance 1, and so on.
     *
     * @param center start node for the BFS; must not be null
     * @param radius maximum hop-distance (must be at least 1)
     * @return a map from each reached Node to its distance from center
     */

    private static Map<Node, Integer> computeDepths(Node center, int radius) {
        Map<Node, Integer> depthMap = new LinkedHashMap<>();
        Queue<Node> queue = new ArrayDeque<>();

        depthMap.put(center, 0);
        queue.add(center);

        while (!queue.isEmpty()) {
            Node current = queue.poll();
            int dist = depthMap.get(current);
            if (dist >= radius) continue;

            for (Node nbr : current.getNeighbors()) {
                if (!depthMap.containsKey(nbr)) {
                    depthMap.put(nbr, dist + 1);
                    queue.add(nbr);
                }
            }
        }
        return depthMap;
    }

    /**
     * Groups node IDs into layers by their exact distance from the centre.
     *
     * The input depths map assigns a distance to each node ID.
     * This method builds a list of radius sets where:
     * - index 0 contains all node IDs at distance 1,
     * - index 1 contains all node IDs at distance 2,
     * - ...
     * - index (radius - 1) contains all node IDs at distance radius.
     *
     * The centre node itself (distance 0) is not included in any layer.
     *
     * @param depths map from node ID to its distance from the centre
     * @param radius maximum distance considered
     * @return an unmodifiable list of radius layers of node IDs
     */

    private static List<Set<Integer>> computeLayers(Map<Integer, Integer> depths, int radius) {
        List<Set<Integer>> layers = new ArrayList<>();
        for (int k = 1; k <= radius; k++) {
            layers.add(new LinkedHashSet<>());
        }
        for (Map.Entry<Integer, Integer> e : depths.entrySet()) {
            int id = e.getKey(), d = e.getValue();
            if (d >= 1 && d <= radius) {
                layers.get(d - 1).add(id);
            }
        }
        return Collections.unmodifiableList(layers);
    }

    /**
     * Prints each extracted Pattern in a human-readable format.
     *
     * @param graph  the input graph
     * @param radius maximum hop-distance
     */
    public static void printPatterns(Graph graph, int radius) {
        extractPatterns(graph, radius).forEach(p -> {
            System.out.println(p);
            System.out.println();
        });
    }

    /* =======================================================================
     * Root-preserving, label-preserving isomorphism for Patterns
     * ==================================================================== */

    /**
     * Checks whether two Patterns are isomorphic as rooted, vertex-labelled
     * ego-networks.
     *
     * The isomorphism must:
     * - map the root of the first pattern to the root of the second,
     * - preserve the depth of every vertex from the root,
     * - preserve the label of every vertex,
     * - preserve adjacency (an edge exists between two vertices in one
     * pattern if and only if the corresponding edge exists between
     * their images in the other pattern).
     *
     * Returns true if such a bijection exists, false otherwise.
     */
    private static boolean areRootedIsomorphic(Pattern a, Pattern b) {
        // Quick structural checks
        if (a.getRadius() != b.getRadius()) return false;

        Map<Integer, Integer> depthsA = a.getDepths();
        Map<Integer, Integer> depthsB = b.getDepths();
        Map<Integer, Integer> labelsA = a.getLabels();
        Map<Integer, Integer> labelsB = b.getLabels();
        Map<Integer, List<Integer>> adjListA = a.getAdjacency();
        Map<Integer, List<Integer>> adjListB = b.getAdjacency();

        if (depthsA.size() != depthsB.size()) return false;
        if (labelsA.size() != labelsB.size()) return false;

        int rootA = a.getId();
        int rootB = b.getId();

        Integer depthRootA = depthsA.get(rootA);
        Integer depthRootB = depthsB.get(rootB);
        if (depthRootA == null || depthRootB == null) return false;
        if (depthRootA != 0 || depthRootB != 0) return false;

        Integer labelRootA = labelsA.get(rootA);
        Integer labelRootB = labelsB.get(rootB);
        if (!Objects.equals(labelRootA, labelRootB)) return false;

        // Check that the multiset of (depth, label) pairs matches.
        if (!sameDepthLabelMultiset(depthsA, labelsA, depthsB, labelsB)) {
            return false;
        }

        // Convert adjacency lists to sets for faster edge checks.
        Map<Integer, Set<Integer>> adjA = toAdjSet(adjListA);
        Map<Integer, Set<Integer>> adjB = toAdjSet(adjListB);

        // Group vertices by (depth, label) and record class of each node.
        Map<DepthLabel, List<Integer>> groupsA = new LinkedHashMap<>();
        Map<Integer, DepthLabel> classA = new HashMap<>();
        buildGroups(depthsA, labelsA, groupsA, classA);

        Map<DepthLabel, List<Integer>> groupsB = new LinkedHashMap<>();
        Map<Integer, DepthLabel> classB = new HashMap<>();
        buildGroups(depthsB, labelsB, groupsB, classB);

        // Root must be in the same depth-label class on both sides.
        DepthLabel rootClassA = classA.get(rootA);
        DepthLabel rootClassB = classB.get(rootB);
        if (!Objects.equals(rootClassA, rootClassB)) return false;

        int totalNodes = depthsA.size();

        Map<Integer, Integer> mapAB = new HashMap<>();
        Map<Integer, Integer> mapBA = new HashMap<>();

        // Fix the root mapping a priori.
        mapAB.put(rootA, rootB);
        mapBA.put(rootB, rootA);

        return backtrackIsomorphism(groupsA, groupsB, classA, adjA, adjB, mapAB, mapBA, totalNodes);
    }

    /**
     * Converts an adjacency list representation into an adjacency set representation.
     *
     * The input map stores, for each node ID, a list of neighbour IDs. This helper
     * wraps each neighbour list in a Hashset so that membership checks
     *
     * @param adj adjacency map from node ID to list of neighbour IDs
     * @return adjacency map from node ID to set of neighbour IDs
     */
    private static Map<Integer, Set<Integer>> toAdjSet(Map<Integer, List<Integer>> adj) {
        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (var e : adj.entrySet()) {
            res.put(e.getKey(), new HashSet<>(e.getValue()));
        }
        return res;
    }

    /**
     * Checks whether two patterns have the same multiset of (depth, label) pairs.
     *
     * For each pattern, this method builds a histogram that counts how many vertices
     * fall into each DepthLabel class (same depth and same label). If the histograms
     * are not identical, then no root- and label-preserving isomorphism can exist,
     * because any bijection would have to preserve both depth and label.
     *
     * This is a cheap necessary condition used to reject impossible matches before
     * running the more expensive backtracking search.
     *
     * @param depthsA map of node ID to depth in pattern A
     * @param labelsA map of node ID to label in pattern A
     * @param depthsB map of node ID to depth in pattern B
     * @param labelsB map of node ID to label in pattern B
     * @return true if the two patterns have identical (depth, label) histograms
     */
    private static boolean sameDepthLabelMultiset(Map<Integer, Integer> depthsA, Map<Integer, Integer> labelsA, Map<Integer, Integer> depthsB, Map<Integer, Integer> labelsB) {
        Map<DepthLabel, Integer> histA = new HashMap<>();
        for (var e : depthsA.entrySet()) {
            int id = e.getKey();
            DepthLabel dl = new DepthLabel(e.getValue(), labelsA.get(id));
            histA.merge(dl, 1, Integer::sum);
        }

        Map<DepthLabel, Integer> histB = new HashMap<>();
        for (var e : depthsB.entrySet()) {
            int id = e.getKey();
            DepthLabel dl = new DepthLabel(e.getValue(), labelsB.get(id));
            histB.merge(dl, 1, Integer::sum);
        }

        return histA.equals(histB);
    }

    /**
     * Builds groups of node IDs keyed by their DepthLabel class and records
     * the class of each node.
     *
     * For each node ID, this method:
     * - looks up its depth and label,
     * - constructs the corresponding DepthLabel,
     * - appends the node ID to the list for that DepthLabel in {@code groups},
     * - stores the DepthLabel in nodeClass so it can be retrieved by ID.
     *
     * These groupings are later used by the isomorphism search to restrict
     * candidate matches to nodes with the same depth and label.
     *
     * @param depths    map of node ID to depth
     * @param labels    map of node ID to label
     * @param groups    output map: DepthLabel -> list of node IDs in that class
     * @param nodeClass output map: node ID -> its DepthLabel class
     */
    private static void buildGroups(Map<Integer, Integer> depths, Map<Integer, Integer> labels, Map<DepthLabel, List<Integer>> groups, Map<Integer, DepthLabel> nodeClass) {
        for (var e : depths.entrySet()) {
            int id = e.getKey();
            int d = e.getValue();
            int lbl = labels.get(id);
            DepthLabel dl = new DepthLabel(d, lbl);
            groups.computeIfAbsent(dl, k -> new ArrayList<>()).add(id);
            nodeClass.put(id, dl);
        }
    }

    /**
     * Backtracking search for a rooted, label-preserving isomorphism between
     * two patterns.
     *
     * The search incrementally builds a partial bijection from vertices of A
     * to vertices of B:
     * - Only vertices in the same DepthLabel class may be matched.
     * - Each candidate match u -> v is checked with respectsAdjacency
     * to ensure that adjacency to already-mapped vertices is preserved.
     * - If a conflict is found, the candidate is rejected and the algorithm
     * backtracks.
     *
     * When the mapping contains totalNodes entries, all vertices have
     * been matched consistently, and the patterns are isomorphic.
     *
     * @param groupsA    vertices of pattern A grouped by DepthLabel
     * @param groupsB    vertices of pattern B grouped by DepthLabel
     * @param classA     map from node ID in A to its DepthLabel
     * @param adjA       adjacency of pattern A (node ID -> neighbour set)
     * @param adjB       adjacency of pattern B (node ID -> neighbour set)
     * @param mapAB      current partial mapping from node IDs in A to node IDs in B
     * @param mapBA      current partial mapping from node IDs in B to node IDs in A
     * @param totalNodes total number of vertices that must be matched
     * @return true if a full adjacency-preserving bijection can be built, false otherwise
     */
    private static boolean backtrackIsomorphism(Map<DepthLabel, List<Integer>> groupsA, Map<DepthLabel, List<Integer>> groupsB, Map<Integer, DepthLabel> classA, Map<Integer, Set<Integer>> adjA, Map<Integer, Set<Integer>> adjB, Map<Integer, Integer> mapAB, Map<Integer, Integer> mapBA, int totalNodes) {
        // All vertices mapped → success.
        if (mapAB.size() == totalNodes) {
            return true;
        }

        // Pick an unmapped vertex u in A.
        int u = -1;
        for (List<Integer> nodes : groupsA.values()) {
            for (int cand : nodes) {
                if (!mapAB.containsKey(cand)) {
                    u = cand;
                    break;
                }
            }
            if (u != -1) break;
        }

        if (u == -1) {
            // Should not happen if totalNodes is consistent, but treat as success.
            return true;
        }

        DepthLabel dl = classA.get(u);
        List<Integer> candidatesB = groupsB.get(dl);
        if (candidatesB == null) {
            return false;
        }

        for (int v : candidatesB) {
            if (mapBA.containsKey(v)) {
                continue; // already used
            }

            if (!respectsAdjacency(u, v, adjA, adjB, mapAB, mapBA)) {
                continue;
            }

            // Tentatively map u -> v
            mapAB.put(u, v);
            mapBA.put(v, u);

            if (backtrackIsomorphism(groupsA, groupsB, classA, adjA, adjB, mapAB, mapBA, totalNodes)) {
                return true;
            }

            // backtrack
            mapAB.remove(u);
            mapBA.remove(v);
        }

        return false;
    }

    /**
     * Checks whether extending the current partial mapping with a new pair
     * u -> v preserves adjacency with all already-mapped vertices.
     *
     * For every mapped vertex x in pattern A with image y = mapAB.get(x),
     * this method verifies that:
     * - there is an edge between u and x in A if and only if
     * - there is an edge between v and y in B.
     *
     * If any such pair violates this condition, the candidate mapping u -> v
     * is inconsistent and must be rejected.
     *
     * @param u     candidate node ID in pattern A
     * @param v     candidate node ID in pattern B
     * @param adjA  adjacency of pattern A (node ID -> neighbour set)
     * @param adjB  adjacency of pattern B (node ID -> neighbour set)
     * @param mapAB current partial mapping from A to B
     * @param mapBA current partial mapping from B to A (unused here but kept for symmetry with the caller)
     * @return true if adjacency is preserved for all already-mapped vertices, false otherwise
     */
    private static boolean respectsAdjacency(int u, int v, Map<Integer, Set<Integer>> adjA, Map<Integer, Set<Integer>> adjB, Map<Integer, Integer> mapAB, Map<Integer, Integer> mapBA) {
        Set<Integer> nbrsA = adjA.getOrDefault(u, Collections.emptySet());
        Set<Integer> nbrsB = adjB.getOrDefault(v, Collections.emptySet());

        for (var e : mapAB.entrySet()) {
            int x = e.getKey();
            int y = e.getValue();

            boolean edgeAX = nbrsA.contains(x);
            boolean edgeBY = nbrsB.contains(y);

            if (edgeAX != edgeBY) {
                return false;
            }
        }
        return true;
    }

    /**
     * Helper value type used to group vertices by their depth and label.
     *
     * During the isomorphism check, two vertices are only allowed to match
     * if they belong to the same (depth, label) class. Grouping by this pair
     * sharply reduces the search space before checking adjacency.
     */

    private static final class DepthLabel {
        final int depth;
        final int label;

        DepthLabel(int depth, int label) {
            this.depth = depth;
            this.label = label;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof DepthLabel)) return false;
            DepthLabel other = (DepthLabel) o;
            return depth == other.depth && label == other.label;
        }

        @Override
        public int hashCode() {
            return Objects.hash(depth, label);
        }
    }
}
