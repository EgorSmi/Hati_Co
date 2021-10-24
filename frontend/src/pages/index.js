import {FinderPage} from "./finder_page/FinderPage"

const PAGES = [
    {
        name: 'FinderPage',
        path: '/'
    },
]

export const PAGES_COMPONENTS = {
    'FinderPage': <FinderPage/>,
}

export default Object.freeze(PAGES)