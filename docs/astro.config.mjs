import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Harness Optimizer',
      description: 'A framework for optimizing LLM agent context through Formulas.',
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        {
          label: 'User Guide',
          items: [
            { label: 'Formulas', link: '/user-guide/formulas/' },
            { label: 'Adapters', link: '/user-guide/adapters/' },
            { label: 'Rewards', link: '/user-guide/rewards/' },
            {
              label: 'Optimizers',
              items: [
                { label: 'Overview', link: '/user-guide/optimizers/' },
                { label: 'Contrastive Reflection', link: '/user-guide/optimizers/contrastive-reflection/' },
              ],
            },
            { label: 'Training', link: '/user-guide/training/' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Formulas', autogenerate: { directory: 'api/formulas' } },
            { label: 'Adapters', autogenerate: { directory: 'api/adapters' } },
            { label: 'Rewards', autogenerate: { directory: 'api/rewards' } },
            { label: 'Optimizers', autogenerate: { directory: 'api/optimizers' } },
            { label: 'Rollout Engines', autogenerate: { directory: 'api/rollout_engines' } },
            { label: 'Data', autogenerate: { directory: 'api/data' } },
          ],
        },
        {
          label: 'Resources',
          items: [
            { label: 'Roadmap', link: '/roadmap/' },
          ],
        },
      ],
    }),
  ],
});
