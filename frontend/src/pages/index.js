import {FinderPage} from "./finder_page/FinderPage"
import {AboutPage} from "./about_page/AboutPage"


const PAGES = [
    {
        name: 'FinderPage',
        path: '/'
    },
    {
        name: "AboutPage",
        path: '/about_us'
    }
]



export const PAGES_COMPONENTS = {
    'FinderPage': <FinderPage/>,
    'AboutPage': <AboutPage/>
}

export default Object.freeze(PAGES)