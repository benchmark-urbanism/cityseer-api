<template lang="pug">
Layout
  VueRemarkContent
</template>

<!-- Front-matter fields can be queried from GraphQL layer -->
<page-query>
query ($id: ID!) {
  doc (id: $id) {
    title
  }
}
</page-query>
