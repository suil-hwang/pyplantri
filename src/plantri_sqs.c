#ifndef PYPLANTRI_SQS_PLUGIN_PASS
#define PYPLANTRI_SQS_PLUGIN_PASS
/* plantri includes PLUGIN after declaring its private graph state. */
#define PLUGIN "plantri_sqs.c"
#include "plantri.c"
#else

/* Fixed source-order records for pyplantri's quartic dual enumerator. */
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define FILTER sqs_filter
#define PLUGIN_INIT sqs_plugin_init()
#define SUMMARY() (dosummary = 0)
#define SQS_FAIL_IF(condition) do { if (condition) exit(1); } while (0)

static void
sqs_order_primal_darts(
    EDGE *ordered_primal_darts[MAXE],
    unsigned short dual_dart_by_primal_dart[NUMEDGES])
{
    EDGE *run, *givenedge, *startedge[MAXN];
    unsigned char neighbor_stamp[MAXN], number[MAXN];
    int actual_number, edge_index, last_number, star_degree, vertex;

    /* Reproduce compute_code()'s rooted BFS dart order while auditing primal stars. */
    memset(neighbor_stamp, 0, (size_t)nv * sizeof(neighbor_stamp[0]));
    memset(number, 0, (size_t)nv * sizeof(number[0]));
    /* The root dart fixes BFS vertices 1 and 2, matching compute_code(). */
    givenedge = code_edge == NULL ? firstedge[0] : code_edge;
    SQS_FAIL_IF(givenedge->start < 0 || givenedge->start >= nv ||
                givenedge->end < 0 || givenedge->end >= nv ||
                givenedge->start == givenedge->end);
    startedge[0] = givenedge;
    number[givenedge->start] = 1;
    number[givenedge->end] = 2;
    startedge[1] = givenedge->invers;
    last_number = 2;
    edge_index = 0;

    /* Preserve each numbered vertex's cyclic dart order in the flat output. */
    for (actual_number = 0; actual_number < last_number; ++actual_number)
    {
        run = startedge[actual_number];
        vertex = run->start;
        SQS_FAIL_IF(vertex < 0 || vertex >= nv ||
                    number[vertex] != actual_number + 1);
        star_degree = 0;
        do
        {
            SQS_FAIL_IF(edge_index >= ne);
            SQS_FAIL_IF(run->start != vertex ||
                        run->end < 0 || run->end >= nv || run->end == vertex ||
                        run->next->prev != run || run->prev->next != run ||
                        run->invers == run || run->invers->invers != run ||
                        run->invers->start != run->end ||
                        run->invers->end != vertex ||
                        neighbor_stamp[run->end] == (unsigned char)(vertex + 1));
            /* Vertex stamps reject parallel neighbors without per-star clears. */
            neighbor_stamp[run->end] = (unsigned char)(vertex + 1);
            ordered_primal_darts[edge_index++] = run;
            dual_dart_by_primal_dart[run - edges] = USHRT_MAX;
            if (!number[run->end])
            {
                SQS_FAIL_IF(last_number >= nv);
                number[run->end] = (unsigned char)++last_number;
                startedge[last_number - 1] = run->invers;
            }
            run = run->next;
            ++star_degree;
            SQS_FAIL_IF(star_degree > degree[vertex]);
        }
        while (run != startedge[actual_number]);
        SQS_FAIL_IF(star_degree != degree[vertex]);
    }
    /* Complete traversal certifies connectedness and the directed-edge count. */
    SQS_FAIL_IF(edge_index != ne || last_number != nv);
}

static void
sqs_plugin_init(void)
{
    /* Accept only -q -c2, with optional -m2. */
    SQS_FAIL_IF(!qswitch || hswitch || Qswitch || oswitch || dswitch ||
                Gswitch || Vswitch || aswitch || gswitch || sswitch ||
                Eswitch || Tswitch || uswitch || vswitch || xswitch ||
                pswitch || bswitch || Aswitch || tswitch || zeroswitch ||
                oneswitch || Xswitch || maxfacesize != -1 ||
                polygonsize != -1 || edgebound[0] != -1 ||
                edgebound[1] != -1 || minconnec != 2 ||
                (minimumdeg != -1 && minimumdeg != 2));
    SQS_FAIL_IF(mod != 1 || res != 0 || outfilename != NULL);

    /* Stay below the bundled min-degree-3 P2/P3 template boundary. */
    SQS_FAIL_IF(UCHAR_MAX != 255 || maxnv < 5 || maxnv >= MAXN ||
                maxnv > (UCHAR_MAX + 1) / 4 + 2);
#ifdef _WIN32
    SQS_FAIL_IF(_setmode(_fileno(stdout), _O_BINARY) == -1);
#endif
    uswitch = TRUE;  /* Suppress the stock writer; FILTER owns stdout. */
}

static int
sqs_filter(int nbtot, int nbop, int doflip)
{
    unsigned char output_record[5 * MAXN - 8];
    unsigned char *dual_twin, *primal_degree_profile;
    unsigned char seen_dual_dart[UCHAR_MAX + 1];
    unsigned short dual_dart_by_primal_dart[NUMEDGES];
    int primal_degree_count[MAXN], dual_face_degree_count[MAXN];
    int primal_face_vertex[4];
    EDGE *ordered_primal_darts[MAXE], *run, *primal_face_start;
    int dual_face_count, dual_face_degree, next_dual_dart;
    int primal_dart, dual_dart, i, j, k;
    size_t record_size;

    /* FILTER output does not depend on automorphism or mirror metadata. */
    (void)nbtot;
    (void)nbop;
    (void)doflip;

    SQS_FAIL_IF(nv != maxnv || ne != 4 * nv - 8 ||
                ne > UCHAR_MAX + 1);
    dual_twin = output_record;
    primal_degree_profile = output_record + ne;
    record_size = (size_t)(ne + nv);
    memset(primal_degree_count, 0,
           (size_t)nv * sizeof(primal_degree_count[0]));
    for (i = 0; i < nv; ++i)
    {
        SQS_FAIL_IF(degree[i] < minimumdeg || degree[i] >= nv);
        ++primal_degree_count[degree[i]];
    }

    /* Integrate check_it(0,4)'s relevant topology checks into this pass. */
    sqs_order_primal_darts(ordered_primal_darts, dual_dart_by_primal_dart);

    /* Each right-face orbit is one quartic dual vertex. */
    dual_dart = 0;
    for (primal_dart = 0; primal_dart < ne; ++primal_dart)
    {
        run = ordered_primal_darts[primal_dart];
        if (dual_dart_by_primal_dart[run - edges] != USHRT_MAX) continue;
        primal_face_start = run;
        for (i = 0; i < 4; ++i)
        {
            for (j = 0; j < i; ++j)
                SQS_FAIL_IF(primal_face_vertex[j] == run->start);
            primal_face_vertex[i] = run->start;
            SQS_FAIL_IF(dual_dart_by_primal_dart[run - edges] != USHRT_MAX);
            dual_dart_by_primal_dart[run - edges] = (unsigned short)dual_dart++;
            run = run->invers->prev;
        }
        SQS_FAIL_IF(run != primal_face_start);
    }
    /* Complete 4-cycle coverage and ne=4*nv-8 give Euler characteristic 2. */
    SQS_FAIL_IF(dual_dart != ne);

    /* Transfer the primal dart involution to the dual twin encoding. */
    for (primal_dart = 0; primal_dart < ne; ++primal_dart)
    {
        run = ordered_primal_darts[primal_dart];
        i = (int)dual_dart_by_primal_dart[run - edges];
        j = (int)dual_dart_by_primal_dart[run->invers - edges];
        SQS_FAIL_IF(i >= ne || j >= ne);
        dual_twin[i] = (unsigned char)j;
    }
    for (dual_dart = 0; dual_dart < ne; ++dual_dart)
    {
        j = (int)dual_twin[dual_dart];
        SQS_FAIL_IF(j >= ne || j == dual_dart ||
                    dual_twin[j] != dual_dart);
    }

    /* The fixed quartic blocks and twin must reproduce the primal profile. */
    memset(seen_dual_dart, 0, (size_t)ne * sizeof(seen_dual_dart[0]));
    memset(dual_face_degree_count, 0,
           (size_t)nv * sizeof(dual_face_degree_count[0]));
    dual_face_count = 0;
    for (i = 0; i < ne; ++i)
    {
        if (seen_dual_dart[i]) continue;
        dual_dart = i;
        dual_face_degree = 0;
        do
        {
            SQS_FAIL_IF(dual_dart < 0 || dual_dart >= ne ||
                        seen_dual_dart[dual_dart]);
            seen_dual_dart[dual_dart] = 1;
            ++dual_face_degree;
            j = (int)dual_twin[dual_dart];
            next_dual_dart = 4 * (j / 4) + (j % 4 + 3) % 4;
            dual_dart = next_dual_dart;
        }
        while (dual_dart != i);
        SQS_FAIL_IF(dual_face_degree < 2 || dual_face_degree >= nv);
        ++dual_face_degree_count[dual_face_degree];
        ++dual_face_count;
    }
    SQS_FAIL_IF(dual_face_count != nv);

    /* Emit the verified degree multiset without reordering dual darts. */
    k = 0;
    for (i = nv - 1; i >= 2; --i)
    {
        SQS_FAIL_IF(primal_degree_count[i] != dual_face_degree_count[i]);
        for (j = 0; j < primal_degree_count[i]; ++j)
            primal_degree_profile[k++] = (unsigned char)i;
    }
    SQS_FAIL_IF(k != nv);

    /* Emit one fixed record; its stream ordinal is the namespace-local Graph ID. */
    SQS_FAIL_IF(fwrite(output_record, 1, record_size, outfile) != record_size);
    return 1;
}

#undef SQS_FAIL_IF
#endif
