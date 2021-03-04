<template lang="pug">
Layout
  VueRemarkContent
</template>

<!-- Front-matter fields can be queried from GraphQL layer -->
<page-query>
query ($id: ID!) {
  doc (id: $id) {
    title
    content
  }
}
</page-query>

<script>
export default {
  name: 'Doc',
  metaInfo() {
    let excerpt = this.$page.doc.content.substring(0, 200)
    excerpt = excerpt.replace('<p>', '').replace('</p>', '')
    return {
      title: `${this.$page.doc.title}`,
      meta: [
        {
          key: 'description',
          name: 'description',
          content: excerpt,
        },
      ],
    }
  },
}
</script>
