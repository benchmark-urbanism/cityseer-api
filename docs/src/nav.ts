// Site section model: four sections rendered in a single sidebar. Section headers are
// always visible; the active section reveals its grouped contents.

export interface NavItem {
  path: string
  label: string
}

export interface NavGroup {
  title?: string
  items: NavItem[]
}

export interface Section {
  id: string
  label: string
  base: string
  groups: NavGroup[]
}

export const sections: Section[] = [
  {
    id: 'start',
    label: 'Getting Started',
    base: '/start',
    groups: [
      {
        items: [{ path: '/start', label: 'Overview' }],
      },
      {
        title: 'Python 101',
        items: [
          { path: '/start/1-notebooks', label: '1 · Notebooks' },
          { path: '/start/2-basics', label: '2 · Python Basics' },
          { path: '/start/3-spatial', label: '3 · Spatial Data' },
          { path: '/start/4-geopandas', label: '4 · GeoPandas' },
          { path: '/start/5-urban', label: '5 · Urban Analytics' },
          { path: '/start/6-data-science', label: '6 · Data Science' },
        ],
      },
    ],
  },
  {
    id: 'guide',
    label: 'Guide',
    base: '/guide/fundamentals',
    groups: [
      {
        items: [
          { path: '/guide/fundamentals', label: 'Fundamentals' },
          { path: '/guide/networks', label: 'Networks' },
          { path: '/guide/cleaning', label: 'Network Cleaning' },
          { path: '/guide/centrality', label: 'Centrality' },
          { path: '/guide/flows', label: 'Origin-Destination Flows' },
          { path: '/guide/land-use', label: 'Land-Use' },
          { path: '/guide/interpretation', label: 'Interpretation' },
          { path: '/guide/troubleshooting', label: 'Troubleshooting' },
          { path: '/guide/migration', label: 'v4 to v5 Migration' },
          { path: '/plugin', label: 'QGIS Plugin' },
        ],
      },
    ],
  },
  {
    id: 'examples',
    label: 'Examples',
    base: '/examples',
    groups: [
      {
        items: [{ path: '/examples', label: 'Overview' }],
      },
      {
        title: 'Recipes',
        items: [
          { path: '/examples/networks', label: 'Network Preparation' },
          { path: '/examples/centrality', label: 'Centrality' },
          { path: '/examples/flows', label: 'Origin-Destination Flows' },
          { path: '/examples/accessibility', label: 'Accessibility' },
          { path: '/examples/stats', label: 'Statistics' },
          { path: '/examples/visibility', label: 'Visibility' },
          { path: '/examples/continuity', label: 'Continuity' },
        ],
      },
      {
        items: [{ path: '/examples/datasets', label: 'Datasets' }],
      },
    ],
  },
  {
    id: 'api',
    label: 'API Reference',
    base: '/api/network',
    groups: [
      {
        title: 'High-level',
        items: [
          { path: '/api/network', label: 'CityNetwork' },
          { path: '/api/decay', label: 'decay' },
        ],
      },
      {
        title: 'Tools',
        items: [
          { path: '/tools/io', label: 'io' },
          { path: '/tools/graphs', label: 'graphs' },
          { path: '/tools/plot', label: 'plot' },
          { path: '/tools/mock', label: 'mock' },
          { path: '/tools/util', label: 'util' },
        ],
      },
      {
        title: 'Metrics',
        items: [
          { path: '/metrics/networks', label: 'networks' },
          { path: '/metrics/layers', label: 'layers' },
          { path: '/metrics/observe', label: 'observe' },
          { path: '/metrics/visibility', label: 'visibility' },
          { path: '/metrics/sampling', label: 'sampling' },
        ],
      },
      {
        title: 'Rust Algos',
        items: [
          { path: '/rustalgos/rustalgos', label: 'rustalgos' },
          { path: '/rustalgos/graph', label: 'graph' },
          { path: '/rustalgos/centrality', label: 'centrality' },
          { path: '/rustalgos/diversity', label: 'diversity' },
          { path: '/rustalgos/data', label: 'data' },
          { path: '/rustalgos/viewshed', label: 'viewshed' },
        ],
      },
      {
        items: [{ path: '/api/glossary', label: 'Glossary' }],
      },
    ],
  },
]

const prefixToSection: [string, string][] = [
  ['/start', 'start'],
  ['/guide', 'guide'],
  ['/plugin', 'guide'],
  ['/examples', 'examples'],
  ['/api', 'api'],
  ['/tools', 'api'],
  ['/metrics', 'api'],
  ['/rustalgos', 'api'],
]

export function resolveSection(path: string): Section {
  const clean = path.replace(/\/$/, '') || '/'
  const match = prefixToSection.find(([prefix]) => clean === prefix || clean.startsWith(`${prefix}/`))
  const id = match ? match[1] : 'start'
  return sections.find((s) => s.id === id) as Section
}
