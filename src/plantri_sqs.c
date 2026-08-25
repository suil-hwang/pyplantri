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

static void compute_code(unsigned char code[]);

static int sqs_require_primal_min_degree_three;

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

    /* Record omitted -m2 before plantri normalizes the default minimum degree. */
    sqs_require_primal_min_degree_three = minimumdeg == -1;
#ifdef _WIN32
    SQS_FAIL_IF(_setmode(_fileno(stdout), _O_BINARY) == -1);
#endif
    uswitch = TRUE;  /* Suppress the stock writer; FILTER owns stdout. */
}

static int
sqs_filter(int nbtot, int nbop, int doflip)
{
    unsigned char primal_planar_code[MAXE + MAXN + 1];
    unsigned char output_record[MAXE + MAXN];
    unsigned char dual_edge_multiplicity[MAXN * MAXN];
    unsigned char *dual_twin, *primal_degree_profile;
    int first_primal_edge_of_vertex[MAXN + 1];
    int primal_edge_start[MAXE], primal_edge_end[MAXE];
    int primal_inverse_edge[MAXE], dual_dart_from_primal_edge[MAXE];
    int primal_edge_by_endpoints[MAXN][MAXN];
    int primal_vertex_seen[MAXN], primal_vertex_stack[MAXN];
    int primal_vertex_count, dual_vertex_count, code_index;
    int primal_directed_edge_count, primal_stack_size, primal_seen_count;
    int u, v, primal_edge, primal_inverse_edge_index, primal_face_start_edge;
    int primal_face_count, dual_dart, dual_twin_dart, dual_dart_count;
    int primal_face_size, dual_u, dual_v, pair_index;
    int dual_double_edge_count, primal_degree_two_vertex_count, i, j;
    size_t record_size;

    /* FILTER output does not depend on automorphism or mirror metadata. */
    (void)nbtot;
    (void)nbop;
    (void)doflip;

    /* Encode the primal quadrangulation in plantri's stored rotation order. */
    compute_code(primal_planar_code);
    primal_vertex_count = primal_planar_code[0];
    dual_vertex_count = primal_vertex_count - 2;
    SQS_FAIL_IF(primal_vertex_count != nv || primal_vertex_count < 5 ||
                ne != 4 * primal_vertex_count - 8 || dual_vertex_count < 3 ||
                4 * dual_vertex_count > 256);

    dual_twin = output_record;
    primal_degree_profile = output_record + 4 * dual_vertex_count;
    record_size = (size_t)(4 * dual_vertex_count + primal_vertex_count);
    for (u = 0; u < primal_vertex_count; ++u)
        for (v = 0; v < primal_vertex_count; ++v)
            primal_edge_by_endpoints[u][v] = -1;

    code_index = 1;
    primal_directed_edge_count = 0;
    first_primal_edge_of_vertex[0] = 0;
    /* Decode directed edges and record primal degrees (= dual face sizes). */
    for (u = 0; u < primal_vertex_count; ++u)
    {
        while (primal_planar_code[code_index] != 0)
        {
            v = (int)primal_planar_code[code_index++] - 1;
            SQS_FAIL_IF(v < 0 || v >= primal_vertex_count || v == u ||
                        primal_edge_by_endpoints[u][v] != -1);
            primal_edge_by_endpoints[u][v] = primal_directed_edge_count;
            primal_edge_start[primal_directed_edge_count] = u;
            primal_edge_end[primal_directed_edge_count++] = v;
        }
        ++code_index;
        first_primal_edge_of_vertex[u + 1] = primal_directed_edge_count;
        primal_degree_profile[u] = (unsigned char)
            (first_primal_edge_of_vertex[u + 1] -
             first_primal_edge_of_vertex[u]);
        SQS_FAIL_IF(sqs_require_primal_min_degree_three &&
                    primal_degree_profile[u] < 3);
    }
    SQS_FAIL_IF(primal_directed_edge_count != ne ||
                code_index != ne + primal_vertex_count + 1);

    /* Pair reciprocal directed primal edges as plantri EDGE inverses. */
    for (primal_edge = 0; primal_edge < primal_directed_edge_count;
         ++primal_edge)
    {
        u = primal_edge_start[primal_edge];
        v = primal_edge_end[primal_edge];
        primal_inverse_edge_index = primal_edge_by_endpoints[v][u];
        SQS_FAIL_IF(primal_inverse_edge_index < 0);
        primal_inverse_edge[primal_edge] = primal_inverse_edge_index;
    }

    /* Primal connectivity is required before face orbits define a dual sphere. */
    memset(primal_vertex_seen, 0, sizeof(primal_vertex_seen));
    primal_vertex_seen[0] = 1;
    primal_seen_count = primal_stack_size = 1;
    primal_vertex_stack[0] = 0;
    while (primal_stack_size)
    {
        u = primal_vertex_stack[--primal_stack_size];
        for (primal_edge = first_primal_edge_of_vertex[u];
             primal_edge < first_primal_edge_of_vertex[u + 1]; ++primal_edge)
            if (!primal_vertex_seen[primal_edge_end[primal_edge]])
            {
                primal_vertex_seen[primal_edge_end[primal_edge]] = 1;
                ++primal_seen_count;
                primal_vertex_stack[primal_stack_size++] =
                    primal_edge_end[primal_edge];
            }
    }
    SQS_FAIL_IF(primal_seen_count != primal_vertex_count);

    /* Trace each primal right face in plantri's invers->prev order. */
    for (primal_edge = 0; primal_edge < primal_directed_edge_count;
         ++primal_edge)
        dual_dart_from_primal_edge[primal_edge] = -1;
    primal_face_count = dual_dart_count = 0;
    for (primal_face_start_edge = 0;
         primal_face_start_edge < primal_directed_edge_count;
         ++primal_face_start_edge)
    {
        int primal_face_boundary_vertices[4];
        if (dual_dart_from_primal_edge[primal_face_start_edge] != -1) continue;
        primal_edge = primal_face_start_edge;
        primal_face_size = 0;
        while (dual_dart_from_primal_edge[primal_edge] == -1)
        {
            SQS_FAIL_IF(primal_face_size >= 4);
            dual_dart_from_primal_edge[primal_edge] = dual_dart_count++;
            primal_face_boundary_vertices[primal_face_size++] =
                primal_edge_start[primal_edge];
            primal_inverse_edge_index = primal_inverse_edge[primal_edge];
            v = primal_edge_end[primal_edge];
            primal_edge =
                primal_inverse_edge_index != first_primal_edge_of_vertex[v]
                ? primal_inverse_edge_index - 1
                : first_primal_edge_of_vertex[v + 1] - 1;
        }
        SQS_FAIL_IF(primal_edge != primal_face_start_edge ||
                    primal_face_size != 4);
        for (i = 0; i < 4; ++i)
            for (j = i + 1; j < 4; ++j)
                SQS_FAIL_IF(primal_face_boundary_vertices[i] ==
                            primal_face_boundary_vertices[j]);
        ++primal_face_count;
    }
    SQS_FAIL_IF(primal_face_count != dual_vertex_count ||
                dual_dart_count != 4 * dual_vertex_count);

    /* Transfer primal EDGE inverses to the dual twin involution. */
    for (primal_edge = 0; primal_edge < primal_directed_edge_count;
         ++primal_edge)
    {
        primal_inverse_edge_index = primal_inverse_edge[primal_edge];
        dual_twin[dual_dart_from_primal_edge[primal_edge]] =
            (unsigned char)
                dual_dart_from_primal_edge[primal_inverse_edge_index];
    }

    /* Dual double edges must match degree-2 primal vertices in the profile. */
    memset(dual_edge_multiplicity, 0, sizeof(dual_edge_multiplicity));
    dual_double_edge_count = 0;
    for (dual_dart = 0; dual_dart < 4 * dual_vertex_count; ++dual_dart)
    {
        dual_twin_dart = dual_twin[dual_dart];
        SQS_FAIL_IF(dual_twin_dart == dual_dart ||
                    dual_twin[dual_twin_dart] != dual_dart);
        if (dual_dart > dual_twin_dart) continue;
        dual_u = dual_dart / 4;
        dual_v = dual_twin_dart / 4;
        SQS_FAIL_IF(dual_u == dual_v);
        pair_index = dual_u < dual_v
            ? dual_u * dual_vertex_count + dual_v
            : dual_v * dual_vertex_count + dual_u;
        if (++dual_edge_multiplicity[pair_index] == 2)
            ++dual_double_edge_count;
        else SQS_FAIL_IF(dual_edge_multiplicity[pair_index] > 2);
    }

    primal_degree_two_vertex_count = 0;
    for (i = 0; i < primal_vertex_count; ++i)
        if (primal_degree_profile[i] == 2)
            ++primal_degree_two_vertex_count;
    SQS_FAIL_IF(dual_double_edge_count != primal_degree_two_vertex_count);

    /* Sort only the profile; dual darts keep primal-face discovery order. */
    for (i = 1; i < primal_vertex_count; ++i)
    {
        unsigned char degree = primal_degree_profile[i];
        j = i;
        while (j > 0 && primal_degree_profile[j - 1] < degree)
        {
            primal_degree_profile[j] = primal_degree_profile[j - 1];
            --j;
        }
        primal_degree_profile[j] = degree;
    }

    /* Emit one fixed record; its stream ordinal is the namespace-local Graph ID. */
    SQS_FAIL_IF(fwrite(output_record, 1, record_size, outfile) != record_size);
    return 1;
}

#undef SQS_FAIL_IF
#endif
