// Loader for the generated notebook manifest (docs/src/generated/notebooks.json).
// The manifest is produced by docs/export_notebooks.py at deploy time and is untracked;
// import.meta.glob tolerates its absence so ordinary builds simply emit no notebook pages.

export interface NotebookEntry {
  slug: string
  section: 'learn' | 'examples'
  topic: string
  title: string
  description: string
  html: string
  download: string
  source: string
}

const modules = import.meta.glob('./generated/notebooks.json', { eager: true }) as Record<
  string,
  { default: NotebookEntry[] }
>

const loaded = Object.values(modules)
export const notebooks: NotebookEntry[] = loaded.length ? loaded[0].default : []

export const learnNotebooks = notebooks
  .filter((n) => n.section === 'learn')
  .sort((a, b) => a.slug.localeCompare(b.slug, undefined, { numeric: true }))

export const exampleNotebooks = notebooks.filter((n) => n.section === 'examples')
