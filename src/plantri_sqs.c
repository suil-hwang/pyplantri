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
sqs_order_edges(EDGE *ordered_edge[MAXE], unsigned short dual_dart_by_edge_storage[NUMEDGES])
{
    EDGE *run, *givenedge, *startedge[MAXN];
    unsigned char number[MAXN];
    int actual_number, last_number, edge_index;

    /* Reproduce compute_code()'s rooted BFS order without planar-code bytes. */
    memset(number, 0, sizeof(number));
    givenedge = code_edge == NULL ? firstedge[0] : code_edge;
    startedge[0] = givenedge;
    number[givenedge->start] = 1;
    number[givenedge->end] = 2;
    startedge[1] = givenedge->invers;
    last_number = 2;
    edge_index = 0;

    for (actual_number = 0; actual_number < last_number; ++actual_number)
    {
        run = startedge[actual_number];
        do
        {
            SQS_FAIL_IF(edge_index >= ne);
            ordered_edge[edge_index++] = run;
            dual_dart_by_edge_storage[run - edges] = USHRT_MAX;
            if (!number[run->end])
            {
                SQS_FAIL_IF(last_number >= nv);
                number[run->end] = (unsigned char)++last_number;
                startedge[last_number - 1] = run->invers;
            }
            run = run->next;
        }
        while (run != startedge[actual_number]);
    }
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
    unsigned short dual_dart_by_edge_storage[NUMEDGES];
    EDGE *ordered_edge[MAXE], *run, *face_start;
    int primal_dart, dual_dart, degree_sum, i, j;
    size_t record_size;

    /* FILTER output does not depend on automorphism or mirror metadata. */
    (void)nbtot;
    (void)nbop;
    (void)doflip;

    SQS_FAIL_IF(nv != maxnv || ne != 4 * nv - 8);
    dual_twin = output_record;
    primal_degree_profile = output_record + ne;
    record_size = (size_t)(ne + nv);
    degree_sum = 0;
    for (i = 0; i < nv; ++i)
    {
        SQS_FAIL_IF(degree[i] < minimumdeg || degree[i] >= nv);
        primal_degree_profile[i] = (unsigned char)degree[i];
        degree_sum += degree[i];
    }
    SQS_FAIL_IF(degree_sum != ne);

    /* Order the generator-owned topology and certify the emitted record. */
    sqs_order_edges(ordered_edge, dual_dart_by_edge_storage);

    /* Each right-face orbit is one quartic dual vertex. */
    dual_dart = 0;
    for (primal_dart = 0; primal_dart < ne; ++primal_dart)
    {
        run = ordered_edge[primal_dart];
        if (dual_dart_by_edge_storage[run - edges] != USHRT_MAX) continue;
        face_start = run;
        for (i = 0; i < 4; ++i)
        {
            SQS_FAIL_IF(dual_dart_by_edge_storage[run - edges] != USHRT_MAX);
            dual_dart_by_edge_storage[run - edges] = (unsigned short)dual_dart++;
            run = run->invers->prev;
        }
        SQS_FAIL_IF(run != face_start);
    }
    SQS_FAIL_IF(dual_dart != ne);

    /* Transfer primal EDGE inverses to the dual twin involution. */
    for (primal_dart = 0; primal_dart < ne; ++primal_dart)
    {
        run = ordered_edge[primal_dart];
        SQS_FAIL_IF(dual_dart_by_edge_storage[run - edges] == USHRT_MAX || dual_dart_by_edge_storage[run->invers - edges] == USHRT_MAX);
        dual_twin[dual_dart_by_edge_storage[run - edges]] = (unsigned char)dual_dart_by_edge_storage[run->invers - edges];
    }

    /* Sort only the profile; dual darts keep primal-face discovery order. */
    for (i = 1; i < nv; ++i)
    {
        unsigned char profile_degree = primal_degree_profile[i];
        j = i;
        while (j > 0 && primal_degree_profile[j - 1] < profile_degree)
        {
            primal_degree_profile[j] = primal_degree_profile[j - 1];
            --j;
        }
        primal_degree_profile[j] = profile_degree;
    }

    /* Emit one fixed record; its stream ordinal is the namespace-local Graph ID. */
    SQS_FAIL_IF(fwrite(output_record, 1, record_size, outfile) != record_size);
    return 1;
}

#undef SQS_FAIL_IF
#endif
