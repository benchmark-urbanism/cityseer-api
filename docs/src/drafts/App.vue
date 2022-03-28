<style lang="postcss" scoped>
#go-box {
  @apply w-full flex flex-col items-center;
}
#go-box:hover {
  transform: scale(1.05);
}
#go-button {
  @apply w-20 h-20 flex items-center justify-center transition-all;
  @apply border-2 border-theme bg-light-grey rounded-full shadow text-theme;
}
#nav-side-by-side {
  @apply flex flex-grow;
}
#nav-tree {
  @apply flex-grow w-full flex flex-col items-end;

  background-color: rgba(211, 47, 47, 0.075);
}
#nested-nav-tree {
  @apply flex flex-col items-end;
}
#logo-link {
  /* width and margins set from animations */
  @apply w-full flex items-center justify-end transition-all;
}
#logo-link:hover {
  transform: scale(1.05);
}
#logo-img {
  @apply w-full object-contain transition-all;
}
.nav-link {
  @apply text-base text-right text-theme font-medium px-3 py-3 cursor-pointer leading-none;
}
.nav-link-active,
.nav-link:hover,
.nav-link:active {
  @apply bg-theme text-white;
}
.nav-link-active {
  @apply border-b border-t border-light-grey;
}
.nested-link {
  @apply text-xs py-1 pr-3 cursor-pointer;
  @apply border-theme;
}
.nested-link-active,
.nested-link:hover,
.nested-link:active {
  @apply border-r-3 pr-2;
}
#footer-container {
  @apply self-start text-xxs px-6 py-1;
}
#title {
  @apply py-4 border-l border-theme transition-all;

  writing-mode: vertical-lr;
}
#title-text {
  @apply text-right font-light text-3xl;
}

#back-to-top {
  @apply fixed flex items-center justify-center z-50 top-4 right-4;
  @apply w-16 h-16 rounded-full border-2 border-white shadow-xl;
  @apply bg-theme text-light-grey;
}

@media only screen and (max-width: 958px) {
  .app-container {
    @apply flex-col items-center;
  }
  #go-button {
    @apply w-16 h-16;
  }

  #nav-column {
    @apply w-full min-w-0 relative min-h-0 max-h-full overflow-visible items-start;
    @apply pl-0 border-r-0 border-b border-theme;

    min-height: 50vh;
  }
  #nav-side-by-side {
    @apply w-full;
  }
  #nav-tree {
    @apply flex-auto items-end pr-0;
  }
  #title {
    @apply flex-1 py-4 ml-0;
  }
  #title-text {
    @apply text-2xl;
  }
  .nav-link {
    @apply text-sm text-left;
  }
  .nested-link {
    @apply text-xs;
  }
  #footer-container {
    @apply text-xxs w-full self-start text-left;
  }
  #content-column {
    @apply max-w-full px-2;
  }
}
</style>

.post-info { @apply flex w-full flex-col justify-center py-4 text-center; } .name { @apply text-sm;
} .date { @apply text-sm font-medium; } @media only screen and (max-width: 750px) { .post-info {
@apply py-1; } }
