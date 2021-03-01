<template>
  <Layout>
    <p class="intro">
      {{ $page.doc.excerpt }}
    </p>
    <VueRemarkContent />
  </Layout>
</template>

<!-- Front-matter fields can be queried from GraphQL layer -->
<page-query>
query ($id: ID!) {
  doc (id: $id) {
    title
  }
}
</page-query>
